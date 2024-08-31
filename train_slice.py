# -*- coding: utf-8 -*-
# Time    : 2024/4/25
# Author  : fanc
# File    : train_slice.py
import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
from src.dataloader import split_pandas, my_dataloader, ThreeDSlicesDataset
import time
import json
from utils import AverageMeter as AverageMeter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.loss import FocalLoss
from src.metrics import bbox_3d_iou, center_distance
import torchvision.models as models
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from utils import load_model
from torch.utils.data import DataLoader
from torchvision import transforms
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
        self.summaryWriter = summaryWriter
        self.only_detection = args.only_detection
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

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
        # else:
        #     self.evaluate()

    def self_model(self):
        if self.args.MODEL_WEIGHT:
            self.model = load_model(model=self.model,
                            checkpoint_path=self.args.MODEL_WEIGHT,
                            multi_gpu=torch.cuda.device_count() > 1)
            print('load model weight success!')
        self.model.to(self.device)

    def calculate_metrics(self, pred, label):
        with torch.no_grad():
            true_labels = label if not label.is_cuda else label.cpu()
            pred = pred if not pred.is_cuda else pred.cpu()
            probabilities = torch.sigmoid(pred.cpu())
            _, predicted_labels = torch.max(probabilities, 1)
            acc = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='binary')
            recall = recall_score(true_labels, predicted_labels, average='binary')
            f1 = f1_score(true_labels, predicted_labels, average='binary')
            return acc, precision, recall, f1,
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
        center_loss = self.box_loss(bbox[:, :3], gtbox[:, :3])
        size_loss = self.box_loss(bbox[:, 3:], gtbox[:, 3:])
        box_loss = center_loss * 0.8 + size_loss * 0.2
        if cls:
            focal_loss = self.focal_loss(cls, label)
        else:
            focal_loss = 0
        return box_loss, focal_loss

    def get_meters(self):
        meters = {
            'loss': AverageMeter(),
            'accuracy': AverageMeter(), 'precision': AverageMeter(), 'recall': AverageMeter(), 'f1': AverageMeter()
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
            img = torch.multiply(img, mask)
            dataset = ThreeDSlicesDataset(img[0][0], gtbox[0], label[0])
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
            for slice, exist_label, cls_label in dataloader:
                slice = slice.repeat(1, 3, 1, 1)
                slice, exist_label, cls_label = slice.to(self.device), exist_label.to(self.device), cls_label.to(self.device)
                out = self.model(slice)
                loss = F.cross_entropy(out, exist_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc, precision, recall, f1 = self.calculate_metrics(out, exist_label)
                self.update_meters(
                    [meters[i] for i in meters.keys()],
                    [loss.item(), acc, precision, recall, f1])
            pbar_train.set_postfix({"loss": meters['loss'].avg,
                                    "acc": meters['accuracy'].avg,
                                    "lr": self.optimizer.param_groups[0]['lr']})
        self.print_metrics(meters, prefix=f'Epoch: [{self.epoch}/{self.epochs}]')
        self.log_metrics_to_tensorboard(meters, self.epoch, stage_prefix='Train')
        self.log_metrics_to_tensorboard({'lr':self.optimizer.param_groups[0]['lr']}, self.epoch)
        self.num_params = sum([param.nelement() for param in self.model.parameters()])
        # save
        self.save_checkpoint(meters)

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
        total_probs = []
        total_preds = []
        total_labels = []
        with torch.no_grad():
            pbar_test = tqdm(enumerate(self.test_loader))
            for inx, (patches, bbox, label, clicinal, pids) in pbar_test:
                patches = [i.to(self.device) for i in patches]
                label = label.to(self.device)
                preds_list = []
                pred_bboxes = []
                for patch in tqdm(patches):
                    dataset = ThreeDSlicesDataset(patch[0][0], bbox[0], label[0])
                    dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

                    pred_bbox, cls = self.model(patch)
                    predictions = F.softmax(cls, dim=1)
                    preds_list.append(predictions)
                    pred_bboxes.append(pred_bbox)
                preds = torch.cat(preds_list, dim=0)
                final_prob = preds.mean(dim=0)
                final_pred = final_prob.argmax()

                total_probs.append(final_prob.unsqueeze(0))
                total_preds.append(final_pred.unsqueeze(0))
                total_labels.append(label)
                # if inx == 2:
                #     break

            total_preds = torch.cat(total_preds, dim=0)
            total_probs = torch.cat(total_probs, dim=0)
            total_labels = torch.cat(total_labels, dim=0)
            print(total_probs.shape, total_preds.shape, total_labels.shape)
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
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    # model = DCNet(num_keys=47, num_classes=4, img_size=(128, 128, 128), crop_size=64, model_depth=args.rd, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    train_info, val_info = split_pandas()
    train_loader = my_dataloader(train_info,
                                batch_size=1,
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
    print(args_dict)

    main(opt, path)