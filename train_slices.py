# -*- coding: utf-8 -*-
# Time    : 2023/10/30 20:35
# Author  : fanc
# File    : train_base.py

import warnings
warnings.filterwarnings("ignore")
import logging  # 引入logging模块
import os.path
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from src.dataloader import LungSliceDataset, split_pandas
from torch.utils.data import DataLoader
from src.nets import get_resnet2d
import time
import json
from utils import AverageMeter as AverageMeter
from utils import evaluate_metrics, load_model, setup_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.loss import FocalLoss
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, optimizer, device, train_loader, test_loader, scheduler, args):
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
        # self.loss_function = FocalLoss(alpha=[1-0.67, 1-0.28], device=device)
        self.loss_function = nn.CrossEntropyLoss()
        # self.summaryWriter = summaryWriter


    def __call__(self):
        if self.args.phase == 'train':
            self.logger = setup_logger(self.args.log_dir)
            self.logger.info("Trainer has been set up.")
            for epoch in tqdm(range(self.epochs)):
                start = time.time()
                self.epoch = epoch+1
                self.train_one_epoch()
                self.num_params = sum([param.nelement() for param in self.model.parameters()])
                self.scheduler.step()
                end = time.time()
                self.logger.info("Epoch: {}, train time: {}".format(epoch, end - start))
        else:
            print("Testing on test set:")
            self.evaluate()

    def self_model(self):
        if self.args.MODEL_WEIGHT:
            self.model = load_model(model=self.model,
                            checkpoint_path=self.args.MODEL_WEIGHT,
                            multi_gpu=torch.cuda.device_count() > 1)
            print('load model weight success!')
        self.model.to(self.device)

    def calculate_metrics(self, pred, label):
        with torch.no_grad():
            probabilities = torch.softmax(pred, dim=1)
            _, predicted_labels = torch.max(probabilities, 1)
            true_labels = label if not label.is_cuda else label.cpu()
            acc = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='macro')
            recall = recall_score(true_labels, predicted_labels, average='macro')
            f1 = f1_score(true_labels, predicted_labels, average='macro')

            return acc, precision, recall, f1
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
    def calculate_loss(self, pred, label):
        # criterion = nn.CrossEntropyLoss()
        loss = self.loss_function(pred, label)
        return loss

    def get_meters(self):
        meters = {
            'loss': AverageMeter(),'accuracy': AverageMeter(), 'precision': AverageMeter(), 'recall': AverageMeter(),
            'f1': AverageMeter()
        }
        return meters

    def update_meters(self, meters, values):
        for meter, value in zip(meters, values):
            meter.update(value)

    def reset_meters(self, meters):
        for meter in meters:
            meter.reset()
    def print_metrics(self, meters, prefix=""):
        for meter, value in meters.items():
            self.logger.info(f'{prefix} {meter}: {value.avg:.4f}' if isinstance(value, AverageMeter) else f'{prefix} {meter}: {value:.4f}')
        # metrics_str = ' '.join([f'{k}: {v.avg:.4f}' if isinstance(v, AverageMeter) else f'{k}: {v:.4f}' for k, v in meters.items()])
        # print(f'{prefix} {metrics_str}')
    def train_one_epoch(self):
        self.model.train()
        meters = self.get_meters()
        all_preds = []
        all_labels = []
        class_correct = list(0. for i in range(self.args.num_classes))
        class_total = list(0. for i in range(self.args.num_classes))
        pbar_train = tqdm(enumerate(self.train_loader))
        for inx, (img, label, pid) in pbar_train:
            img, label = img.to(self.device), label.to(self.device)

            cls = self.model(img)
            loss = self.calculate_loss(cls, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_preds.extend(cls.detach().cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            acc, precision, recall, f1 = self.calculate_metrics(cls.cpu(), label.cpu())
            self.update_meters(
                [meters[i] for i in meters.keys()],
                [loss, acc, precision, recall, f1])

            _, predicted = torch.max(cls, 1)
            # Record the correct predictions
            correct = (predicted == label).squeeze()

            if correct.ndim == 0:
                correct = correct.unsqueeze(0)
            for i in range(label.size(0)):
                label_i = label[i].item()
                class_correct[label_i] += correct[i].item()
                class_total[label_i] += 1
            # print(f'data load time: {time.time() - start_time}')
            acc_class = [i / (j + 10e-6) for i, j in zip(class_correct, class_total)]
            pbar_train.set_postfix({"loss": loss.item(), "acc": acc, "acc_class": acc_class})

        class_accuracies = {}
        for i in range(self.args.num_classes):
            if class_total[i] > 0:
                class_accuracies[i] = 100 * class_correct[i] / class_total[i]
            else:
                class_accuracies[i] = 0  # No samples for this class in the dataset
        self.logger.info("Train Accuracy per class:", class_accuracies)

        meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], meters['auc'] = self.calculate_all_metrics(all_preds, all_labels)
        meters['lr'] = self.optimizer.param_groups[0]['lr']
        self.print_metrics(meters, prefix='Train')
        self.save_checkpoint(meters)
        if self.epoch % 10 == 0:
            self.evaluate()

    def save_checkpoint(self, meters):
        if self.args.phase == 'train':
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
                }, os.path.join(self.args.save_dir, 'best_checkpoint.pt'))
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
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'model_last.pt'))
                print(f"New checkpoint saved at epoch {self.epoch} with accuracy: {meters['accuracy']:.4f}")
    def evaluate(self):
        self.model.eval()
        meters = self.get_meters()
        all_preds = []
        all_labels = []
        class_correct = list(0. for i in range(self.args.num_classes))
        class_total = list(0. for i in range(self.args.num_classes))
        with torch.no_grad():
            pbar_test = tqdm(enumerate(self.test_loader))
            for inx, (img, label, pids) in pbar_test:
                img, label = img.to(self.device), label.to(self.device)
                cls = self.model(img)
                loss = self.calculate_loss(cls, label)
                all_preds.extend(cls.detach().cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                acc, precision, recall, f1 = self.calculate_metrics(cls.cpu(), label.cpu())
                self.update_meters(
                    [meters[i] for i in meters.keys()],
                    [loss, acc, precision, recall, f1])

                _, predicted = torch.max(cls, 1)
                # Record the correct predictions
                correct = (predicted == label).squeeze()

                if correct.ndim == 0:
                    correct = correct.unsqueeze(0)
                for i in range(label.size(0)):
                    label_i = label[i].item()
                    class_correct[label_i] += correct[i].item()
                    class_total[label_i] += 1
                # print(f'data load time: {time.time() - start_time}')
                acc_class = [i/(j+10e-6) for i, j in zip(class_correct, class_total)]
                pbar_test.set_postfix({"loss": loss.item(), "acc": acc, "acc_class": acc_class})

            class_accuracies = {}
            for i in range(self.args.num_classes):
                if class_total[i] > 0:
                    class_accuracies[i] = class_correct[i] / class_total[i]
                else:
                    class_accuracies[i] = 0  # No samples for this class in the dataset

            self.logger.info(f"Val Accuracy per class:{class_accuracies}")
            meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], meters[
                'auc'] = self.calculate_all_metrics(all_preds, all_labels)
            meters['lr'] = self.optimizer.param_groups[0]['lr']
            self.save_checkpoint(meters)
            self.print_metrics(meters, prefix='Val')
            self.save_checkpoint(meters)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main(args, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    model = get_resnet2d(rd=args.rd, num_classes=args.num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001, betas=(0.9, 0.99))
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    train_info, val_info = split_pandas()
    train_dataset = LungSliceDataset(train_info, phase='train')
    val_dataset = LungSliceDataset(val_info, phase='val')
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=6
                                    )
    val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=6
                                    )

    if args.phase == 'train':
        log_path = makedirs(os.path.join(path, 'logs'))
        model_path = makedirs(os.path.join(path, 'models'))
        args.log_dir = log_path
        args.save_dir = model_path

    trainer = Trainer(model,
                      optimizer,
                      device,
                      train_loader,
                      val_loader,
                      scheduler,
                      args,
                      )
    trainer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--rd', type=int, default=18)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')

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