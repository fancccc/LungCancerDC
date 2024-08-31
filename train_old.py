# -*- coding: utf-8 -*-
# Time    : 2024/4/21
# Author  : fanc
# File    : train_old.py
import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from src.dataloader import split_pandas, my_dataloader
import time
import json
from utils import AverageMeter as AverageMeter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.loss import FocalLoss, FocalLoss3dMap
from src.metrics import bbox_3d_iou, center_distance, find_max_points
from src.nets import DCNet, DetectionLungCancer, UNETR
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import logging

class Trainer:
    def __init__(self, model, optimizer, device, train_loader, test_loader, scheduler, args, summaryWriter):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = args.epochs
        self.epoch = 0
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.args = args
        self.self_model()
        self.focal_loss = FocalLoss(alpha=[1-0.67, 1-0.28, 1-0.025, 1-0.015], device=device)
        self.box_loss = nn.SmoothL1Loss()
        self.center_map_loss = FocalLoss3dMap()
        self.summaryWriter = summaryWriter
        self.only_detection = args.only_detection

    def __call__(self):
        if self.args.phase == 'train':
            total_start_time = time.time()
            for epoch in range(self.epochs):
                start_time = time.time()
                self.epoch = epoch+1
                self.train_one_epoch()
                self.scheduler.step()
                epoch_time = time.time() - start_time
                total_elapsed_time = time.time() - total_start_time
                estimated_total_time = (total_elapsed_time / (epoch + 1)) * self.epochs
                time_remaining = estimated_total_time - total_elapsed_time
                print("Epoch: {}, train time: {:.2f} sec, params number: {}, ".format(epoch + 1, epoch_time,
                                                                                      self.num_params) +
                      "estimated completion time: {}, time remaining: {:.2f} sec".format(
                          time.strftime('%m-%d %H:%M', time.localtime(total_start_time + estimated_total_time)),
                          time_remaining))
        else:
            self.evaluate()

    def self_model(self):
        if self.args.MODEL_WEIGHT:
            self.model = load_model(model=self.model,
                            checkpoint_path=self.args.MODEL_WEIGHT,
                            multi_gpu=torch.cuda.device_count() > 1)
            print('load model weight success!')
        self.model.to(self.device)

    def calculate_metrics(self, pred, label, pred_boxes, gt_boxes):
        with torch.no_grad():
            true_labels = label if not label.is_cuda else label.cpu()
            pred_boxes = pred_boxes if pred_boxes.nelement() > 0 else pred_boxes.cpu()
            gt_boxes = gt_boxes if gt_boxes.nelement() > 0 else gt_boxes.cpu()
            if pred:
                pred = pred if not pred.is_cuda else pred.cpu()
                probabilities = torch.softmax(pred.cpu(), dim=1)
                _, predicted_labels = torch.max(probabilities, 1)
                acc = accuracy_score(true_labels, predicted_labels)
                precision = precision_score(true_labels, predicted_labels, average='macro')
                recall = recall_score(true_labels, predicted_labels, average='macro')
                f1 = f1_score(true_labels, predicted_labels, average='macro')
            else:
                acc, precision, recall, f1 = 0, 0, 0, 0
            pred_boxes = find_max_points(torch.sigmoid(pred_boxes))
            IoU = bbox_3d_iou(pred_boxes.cpu(), gt_boxes.cpu())
            # print(pred_boxes.cpu(), gt_boxes.cpu())
            logging.info(f'Predicted bboxes: {pred_boxes.cpu()}, GT: {gt_boxes.cpu()}')
            norm_dist = center_distance(pred_boxes.cpu(), gt_boxes.cpu())
            # auc = roc_auc_score(torch.nn.functional.one_hot(label, num_classes=probabilities.shape[1]), probabilities,
            #                     multi_class='ovr', average='macro')
            return acc, precision, recall, f1, IoU.item(), norm_dist.item()
    def calculate_all_metrics(self, pred, label):
        pred = torch.tensor(pred)
        label = torch.tensor(label)
        probabilities = torch.softmax(pred, dim=1)
        _, predicted_labels = torch.max(probabilities, 1)

        label = label.numpy()
        predicted_labels = predicted_labels.numpy()

        acc = accuracy_score(label, predicted_labels)
        precision = precision_score(label, predicted_labels, average='macro')
        recall = recall_score(label, predicted_labels, average='macro')
        f1 = f1_score(label, predicted_labels, average='macro')
        try:
            auc = roc_auc_score(torch.nn.functional.one_hot(torch.tensor(label), num_classes=probabilities.shape[1]), probabilities.numpy(),
                                multi_class='ovr', average='macro')
        except Exception as e:
            print(e)
            auc = 0
        return acc, precision, recall, f1, auc
    def calculate_loss(self, bbox, cls, label, gtbox):
        center_dist_loss = self.box_loss(find_max_points(bbox)[:, :3], gtbox[:, :3])
        # size_loss = self.box_loss(bbox[:, 3:], gtbox[:, 3:])
        # box_loss = center_loss * 0.8 + size_loss * 0.2
        center_map_loss = self.center_map_loss(torch.sigmoid(bbox), gtbox)
        box_loss = center_dist_loss + center_map_loss
        if cls:
            focal_loss = self.focal_loss(cls, label)
        else:
            focal_loss = 0
        return box_loss, focal_loss

    def get_meters(self):
        meters = {
            'loss': AverageMeter(), 'box_loss': AverageMeter(), 'focal_loss': AverageMeter(),
            'accuracy': AverageMeter(), 'precision': AverageMeter(), 'recall': AverageMeter(), 'f1': AverageMeter(),
            'IoU': AverageMeter(), 'center_dist': AverageMeter(),
        }
        return meters

    def update_meters(self, meters, values):
        for meter, value in zip(meters, values):
            meter.update(value)

    def reset_meters(self, meters):
        for meter in meters:
            meter.reset()
    def print_metrics(self, meters, prefix=""):
        metrics_str = ' '.join([f'{k}: {v.avg:.4f}' if isinstance(v, AverageMeter) else f'{k}: {v:.4f}' for k, v in meters.items()])
        print(f'{prefix} {metrics_str}')
    def log_metrics_to_tensorboard(self, metrics, epoch, stage_prefix=''):
        for name, meter in metrics.items():
            if 'loss' not in name.lower():
                category_prefix = 'Metric'
            else:
                category_prefix = 'Loss'
            tag = f'{category_prefix}/{name}'
            if 'lr' in name.lower():
                tag = 'lr'
            value = meter.avg if isinstance(meter, AverageMeter) else meter
            self.summaryWriter.add_scalars(tag, {stage_prefix: value}, epoch)
    def train_one_epoch(self):
        self.model.train()
        meters = self.get_meters()
        all_preds = []
        all_labels = []
        class_correct = list(0. for i in range(4))
        class_total = list(0. for i in range(4))
        pbar_train = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
        for inx, (img, mask, gtbox, label, clicinal, pid) in pbar_train:
            img, mask, label, gtbox = img.to(self.device), mask.to(self.device), label.to(self.device), gtbox.to(self.device)
            clicinal = clicinal.to(self.device)
            # only lung area
            img = torch.multiply(img, mask)
            pred_bbox, cls = self.model(clicinal, img, self.only_detection)

            box_loss, focal_loss = self.calculate_loss(pred_bbox, cls, label, gtbox)
            if cls:
                loss = 0.5*box_loss + 0.5*focal_loss
            else:
                loss = box_loss
            if torch.isnan(loss).any():
                print(f'pred_bbox: {pred_bbox}, cls: {cls}')
                break
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if cls:
                all_preds.extend(cls.detach().cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                _, predicted = torch.max(cls, 1)
                # Record the correct predictions
                correct = (predicted == label).squeeze()
                # 添加这个检查以确保correct始终是一维的
                if correct.ndim == 0:
                    correct = correct.unsqueeze(0)
                for i in range(label.size(0)):
                    label_i = label[i].item()
                    class_correct[label_i] += correct[i].item()
                    class_total[label_i] += 1

            acc, precision, recall, f1, IoU, center_dist = self.calculate_metrics(cls, label, pred_bbox, gtbox)
            # print(f'box_loss: {box_loss},IoU_loss: {IoU_loss},focal_loss1: {focal_loss1}, focal_loss2: {focal_loss2}')
            self.update_meters(
                [meters[i] for i in meters.keys()],
                [loss.item(), box_loss.item(), focal_loss.item() if focal_loss != 0 else 0, acc, precision, recall, f1, IoU, center_dist])
            pbar_train.set_postfix({"loss": loss.item(),
                                    "acc": acc,
                                    "IoU": IoU,
                                    "center_dist": center_dist,
                                    "lr": self.optimizer.param_groups[0]['lr']})
        # print(f'pred bbox: {pred_bbox.detach().cpu().numpy()}, gtbbox: {gtbox.cpu().numpy()}')

        if not self.only_detection:
            # Calculate accuracy for each class
            class_accuracies = {}
            for i in range(4):
                if class_total[i] > 0:
                    class_accuracies[i] = 100 * class_correct[i] / class_total[i]
                else:
                    class_accuracies[i] = 0  # No samples for this class in the dataset

            # print("Accuracy per class:", class_accuracies)
            self.print_metrics(class_accuracies, prefix=f'Epoch: [{self.epoch}/{self.epochs}]')
            meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], meters['auc'] = self.calculate_all_metrics(all_preds, all_labels)
        self.print_metrics(meters, prefix=f'Epoch: [{self.epoch}/{self.epochs}]')
        self.log_metrics_to_tensorboard(meters, self.epoch, stage_prefix='Train')
        self.log_metrics_to_tensorboard({'lr':self.optimizer.param_groups[0]['lr']}, self.epoch)
        self.num_params = sum([param.nelement() for param in self.model.parameters()])
        # save
        self.save_checkpoint(meters)
        # if meters['IoU'].avg > 0.6:
        #     self.only_dection = False

    def save_checkpoint(self, meters):
        beat_acc = meters['accuracy'].avg if isinstance(meters['accuracy'], AverageMeter) else meters['accuracy']
        if self.args.phase == 'train':
            if beat_acc > self.best_acc:
                self.best_acc = beat_acc
                self.best_acc_epoch = self.epoch
                self.best_metrics = meters
                with open(os.path.join(os.path.dirname(self.args.save_dir), 'best_acc_metrics.json'), 'w') as f:
                    json.dump({k: v for k, v in meters.items() if not isinstance(v, AverageMeter)}, f)

                torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_acc': self.best_acc,
                }, os.path.join(self.args.save_dir, 'best_checkpoint.pth'))
                print(f"New best model saved at epoch {self.best_acc_epoch} with accuracy: {self.best_acc:.4f}")

            if self.epoch % self.args.save_epoch == 0:
                checkpoint = {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_acc': beat_acc,
                    'num_params': self.num_params
                }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-%d.pth' % self.epoch))
                print(f"New checkpoint saved at epoch {self.epoch} with accuracy: {beat_acc:.4f}")
            checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_acc': beat_acc,
                'num_params': self.num_params
            }
            torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-latest.pth'))
            print(f"New checkpoint saved at epoch {self.epoch} with accuracy: {beat_acc:.4f}")

    def evaluate(self):
        self.model.eval()
        pbar_train = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
        results = []
        for inx, (img, mask, gtbox, label, clicinal, pid) in pbar_train:
            img, mask, label, gtbox = img.to(self.device), mask.to(self.device), label.to(self.device), gtbox.to(self.device)
            clicinal = clicinal.to(self.device)
            # only lung area
            img = torch.multiply(img, mask)
            pred_bbox, cls = self.model(clicinal, img, self.only_detection)

            acc, precision, recall, f1, IoU, center_dist = self.calculate_metrics(cls, label, pred_bbox, gtbox)
            pred_bbox = find_max_points(pred_bbox)
            pbar_train.set_postfix({"IoU": IoU, "center_dist": center_dist})
            temp = {'img': img.cpu().numpy().tolist(),
                            'gtbox': gtbox.cpu().numpy().tolist(),
                            'pred_bbox': pred_bbox.cpu().numpy().tolist(),
                            'IoU': IoU, 'center_dist': center_dist, 'pid': pid}
            # for i in temp:
            #     print(type(temp[i]))
            results.append(temp)
            if inx == 5:
                with open('./results/results.json', 'w') as f:
                    json.dump(results, f)
                break


        # total_probs = []
        # total_preds = []
        # total_labels = []
        # with torch.no_grad():
            # pbar_test = tqdm(enumerate(self.test_loader))
            # for inx, (patches, bbox, label, clicinal, pids) in pbar_test:
            #     patches = [i.to(self.device) for i in patches]
            #     label = label.to(self.device)
            #     preds_list = []
            #     pred_bboxes = []
            #     for patch in tqdm(patches):
            #         pred_bbox, cls = self.model(patch)
            #         pred_bboxes.append(pred_bbox)

                    # predictions = F.softmax(cls, dim=1)
                    # preds_list.append(predictions)
                    # pred_bboxes.append(pred_bbox)
                # preds = torch.cat(preds_list, dim=0)
                # final_prob = preds.mean(dim=0)
                # final_pred = final_prob.argmax()
                #
                # total_probs.append(final_prob.unsqueeze(0))
                # total_preds.append(final_pred.unsqueeze(0))
                # total_labels.append(label)
                # if inx == 2:
                #     break

            # total_preds = torch.cat(total_preds, dim=0)
            # total_probs = torch.cat(total_probs, dim=0)
            # total_labels = torch.cat(total_labels, dim=0)
            # print(total_probs.shape, total_preds.shape, total_labels.shape)
            # accuracy = (total_preds == total_labels).float().mean().item() * 100
            # overall_acc, class_acc, recall, precision, f1, auc = evaluate_metrics(total_probs, total_preds, total_labels)
            # print(f'overall_acc: {overall_acc}, class_acc: {class_acc}, recall: {recall},\
            #  precision: {precision}, f1: {f1}, auc: {auc}')

            # self.print_metrics(meters, prefix=f'Epoch-Val: [{self.epoch}/{self.epochs}]')
            # # 更新学习率调度器
            # self.scheduler.step(meters['loss'].avg)
            # # 记录性能指标到TensorBoard
            # self.log_metrics_to_tensorboard(meters, self.epoch, stage_prefix='Val')
            # print(f'Best acc is {self.best_acc} at epoch {self.best_acc_epoch}!')
            # print(f'{self.best_acc}=>{meters["accuracy"]}')


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main(args, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    # model = DCNet(num_keys=47, num_classes=4, img_size=(128, 128, 128), crop_size=64, model_depth=args.rd, device=device)
    # model = DetectionLungCancer(model_depth=args.rd)
    model = UNETR(in_channels=1, out_channels=1, img_size=(128, 128, 128))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    train_info, val_info = split_pandas()
    train_loader = my_dataloader(train_info,
                                batch_size=args.batch_size,
                                shuffle=True,
                                    )
    val_loader = my_dataloader(val_info,
                                batch_size=1,
                                shuffle=False,
                                phase='test'
                                    )
    summaryWriter = None
    if args.phase == 'train':
        log_path = makedirs(os.path.join(path, 'logs'))
        model_path = makedirs(os.path.join(path, 'models'))
        args.log_dir = log_path
        args.save_dir = model_path
        summaryWriter = SummaryWriter(log_dir=args.log_dir)
    trainer = Trainer(model,
                      optimizer,
                      device,
                      train_loader,
                      val_loader,
                      scheduler,
                      args,
                      summaryWriter)
    trainer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rd', type=int, default=50)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--only_detection', type=bool, default=True)


    opt = parser.parse_args()
    args_dict = vars(opt)
    now = time.strftime('%y%m%d%H%M', time.localtime())
    path = None
    if opt.phase == 'train':
        if not os.path.exists(f'./results/{now}'):
            os.makedirs(f'./results/{now}')
        path = f'./results/{now}'
        with open(os.path.join(path, 'train_config.json'), 'w') as fp:
            json.dump(args_dict, fp, indent=4)
        print(f"Training configuration saved to {now}")
        log_dir = f'{path}/logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'training_log.log')
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    print(args_dict)

    main(opt, path)