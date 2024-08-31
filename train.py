# -*- coding: utf-8 -*-
# Time    : 2023/10/30 20:35
# Author  : fanc
# File    : train_base.py

import warnings
warnings.filterwarnings("ignore")
import os.path
import os
import argparse
import pandas as pd
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from src.dataloader import split_pandas, my_dataloader
from torch.nn.parallel import DataParallel
from src.resnet import generate_model
import time
import json
import torch.nn.functional as F
from utils import AverageMeter as AverageMeter
from utils import calculate_acc_sigmoid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.loss import FocalLoss
from src.metrics import bbox_3d_iou
from src.nets import DCNet
def load_model(model, checkpoint_path, multi_gpu=False):
    print("Loading model...")
    pretrain = torch.load(checkpoint_path)
    if 'model_state_dict' in pretrain.keys():
        state_dict = pretrain['model_state_dict']
    else:
        state_dict = pretrain['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    for name, param in model.named_parameters():
        if name in state_dict and param.size() == state_dict[name].size():
            param.data.copy_(state_dict[name])
            # print(f"Loaded layer: {name}")
        else:
            print(f"Skipped layer: {name}")
    if multi_gpu:
        model = nn.DataParallel(model)
    print("Finished loading model!")
    return model

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

    def __call__(self):
        if self.args.phase == 'train':
            for epoch in tqdm(range(self.epochs)):
                start = time.time()
                self.epoch = epoch+1
                self.train_one_epoch()
                self.num_params = sum([param.nelement() for param in self.model.parameters()])
                # self.scheduler.step()
                end = time.time()
                print("Epoch: {}, train time: {}".format(epoch, end - start))
                if epoch % 1 == 0:
                    self.evaluate()
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
            probabilities = torch.softmax(pred, dim=1)
            _, predicted_labels = torch.max(probabilities, 1)
            true_labels = label if not label.is_cuda else label.cpu()
            acc = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='macro')
            recall = recall_score(true_labels, predicted_labels, average='macro')
            f1 = f1_score(true_labels, predicted_labels, average='macro')
            IoU = bbox_3d_iou(pred_boxes, gt_boxes)
            # auc = roc_auc_score(torch.nn.functional.one_hot(label, num_classes=probabilities.shape[1]), probabilities,
            #                     multi_class='ovr', average='macro')

            return acc, precision, recall, f1, IoU.item()
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
        box_loss = self.box_loss(bbox, gtbox)
        focal_loss = self.focal_loss(cls, label)
        return box_loss, focal_loss

    def get_meters(self):
        meters = {
            'loss': AverageMeter(), 'box_loss': AverageMeter(), 'focal_loss': AverageMeter(),
            'accuracy': AverageMeter(), 'precision': AverageMeter(), 'recall': AverageMeter(), 'f1': AverageMeter(),
            'IoU': AverageMeter()
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
        pbar_train = tqdm(enumerate(self.train_loader))
        for inx, (img, label, gtbox, clicinal) in pbar_train:
            img, label = img.to(self.device), label.to(self.device)
            gtbox = gtbox.to(self.device)
            clicinal = clicinal.to(self.device)
            pred_bbox, cls = self.model(clicinal, img)
            box_loss, focal_loss = self.calculate_loss(pred_bbox, cls, label, gtbox)
            loss = 0.5*box_loss + 0.5*focal_loss
            if torch.isnan(loss).any():
                print(f'pred_bbox: {pred_bbox}, cls: {cls}')
                break
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_preds.extend(cls.detach().cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            acc, precision, recall, f1, IoU = self.calculate_metrics(cls.cpu(), label.cpu(), pred_bbox.cpu(), gtbox.cpu())
            # print(f'box_loss: {box_loss},IoU_loss: {IoU_loss},focal_loss1: {focal_loss1}, focal_loss2: {focal_loss2}')
            self.update_meters(
                [meters[i] for i in meters.keys()],
                [loss.item(), box_loss.item(), focal_loss.item(), acc, precision, recall, f1, IoU])

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

            pbar_train.set_postfix({"loss": loss.item(), "acc": acc, "IoU": IoU})
        # Calculate accuracy for each class
        class_accuracies = {}
        for i in range(4):
            if class_total[i] > 0:
                class_accuracies[i] = 100 * class_correct[i] / class_total[i]
            else:
                class_accuracies[i] = 0  # No samples for this class in the dataset

        print("Accuracy per class:", class_accuracies)
        meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], meters['auc'] = self.calculate_all_metrics(all_preds, all_labels)
        self.print_metrics(meters, prefix=f'Epoch: [{self.epoch}/{self.epochs}]')
        self.print_metrics(class_accuracies, prefix=f'Epoch: [{self.epoch}/{self.epochs}]')
        self.log_metrics_to_tensorboard(meters, self.epoch, stage_prefix='Train')
        self.log_metrics_to_tensorboard({'lr':self.optimizer.param_groups[0]['lr']}, self.epoch)

    def evaluate(self):
        self.model.eval()
        meters = self.get_meters()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inx, (img, label, gtbox, clicinal) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                img, label = img.to(self.device), label.to(self.device)
                gtbox = gtbox.to(self.device)
                clicinal = clicinal.to(self.device)
                pred_bbox, cls = self.model(clicinal, img)
                box_loss, focal_loss = self.calculate_loss(pred_bbox, cls, label, gtbox)
                loss = 0.5 * box_loss + 0.5 * focal_loss
                all_preds.extend(cls.detach().cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                acc, precision, recall, f1, IoU = self.calculate_metrics(cls.cpu(), label.cpu(), pred_bbox.cpu(), gtbox.cpu())
                self.update_meters(
                    [meters[i] for i in meters.keys()],
                    [loss.item(), box_loss.item(), focal_loss.item(), acc, precision, recall, f1, IoU])
            meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], meters[
                'auc'] = self.calculate_all_metrics(all_preds, all_labels)
            self.print_metrics(meters, prefix=f'Epoch-Val: [{self.epoch}/{self.epochs}]')
            # 更新学习率调度器
            self.scheduler.step(meters['loss'].avg)
            # 记录性能指标到TensorBoard
            self.log_metrics_to_tensorboard(meters, self.epoch, stage_prefix='Val')
            print(f'Best acc is {self.best_acc} at epoch {self.best_acc_epoch}!')
            print(f'{self.best_acc}=>{meters["accuracy"]}')

            if self.args.phase == 'train':
                # 检查并保存最佳模型
                if meters['accuracy'] > self.best_acc:
                    self.best_acc_epoch = self.epoch
                    self.best_acc = meters['accuracy']
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
                        'model_state_dict': self.model.state_dict(),  # *模型参数
                        'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                        'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                        'best_acc': meters['accuracy'],
                        'num_params': self.num_params
                    }
                    torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-%d.pth' % self.epoch))
                    print(f"New checkpoint saved at epoch {self.epoch} with accuracy: {meters['accuracy']:.4f}")


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main(args, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    model = DCNet(num_keys=47, num_classes=4, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    train_info, val_info = split_pandas()
    train_loader = my_dataloader(train_info,
                                batch_size=args.batch_size,
                                shuffle=True,
                                    )
    val_loader = my_dataloader(val_info,
                                batch_size=args.batch_size,
                                shuffle=False,
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
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--rd', type=int, default=50)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--dropout', type=float, default=0)

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