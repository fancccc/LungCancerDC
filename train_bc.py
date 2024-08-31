# -*- coding: utf-8 -*-
'''
@file: train_bc.py
@author: fanc
@time: 2024/6/11 20:57
'''

import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from src.dataloader import split_pandas, LungDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import time
import json
from src.loss import FocalLoss
from torch.optim.lr_scheduler import ExponentialLR
from src.trainer import BaseTrainer
from utils import makedirs

class BCTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(BCTrainer, self).__init__(*args, **kwargs)
        self.loss_function = FocalLoss(alpha=self.args.loss_weight, device=self.device)
        # weight = torch.tensor([0.64, 0.29]).to(self.device)
        # self.loss_function = torch.nn.CrossEntropyLoss(weight=weight)

    def train_one_epoch(self):
        meters = self.get_meters()
        self.model.train()
        all_preds = []
        all_labels = []
        class_correct = list(0. for i in range(self.args.num_classes))
        class_total = list(0. for i in range(self.args.num_classes))
        pbar_train = tqdm(enumerate(self.train_loader))
        for inx, data in pbar_train:
            ct128, bbox, label = data['ct128'].to(self.device), data['bbox'].to(self.device), data['label'].to(self.device)
            cls = self.model(ct128, bbox)
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
            correct = (predicted == label).squeeze()
            if correct.ndim == 0:
                correct = correct.unsqueeze(0)
            for i in range(label.size(0)):
                label_i = label[i].item()
                class_correct[label_i] += correct[i].item()
                class_total[label_i] += 1
            pbar_train.set_postfix({"loss": loss.item(), "acc": acc})
        # Calculate accuracy for each class
        class_accuracies = self.class_accuracies(class_correct, class_total)
        meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], meters['auc'] = self.calculate_all_metrics(all_preds, all_labels)
        meters['lr'] = self.optimizer.param_groups[0]['lr']
        meters.update(class_accuracies)
        self.print_metrics(meters, prefix=f'Train epoch:{self.epoch}/{self.epochs}')
        if self.epoch % 5 == 0:
            self.evaluate()
        self.save_checkpoint(meters, 'train')

    def evaluate(self):
        self.model.eval()
        meters = self.get_meters()
        all_preds = []
        all_labels = []
        class_correct = list(0. for i in range(self.args.num_classes))
        class_total = list(0. for i in range(self.args.num_classes))
        with torch.no_grad():
            pbar_test = tqdm(enumerate(self.test_loader))
            for inx, data in pbar_test:
                ct128, bbox, label = data['ct128'].to(self.device), data['bbox'].to(self.device), data['label'].to(
                    self.device)
                cls = self.model(ct128, bbox)
                loss = self.calculate_loss(cls, label)
                all_preds.extend(cls.detach().cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                acc, precision, recall, f1 = self.calculate_metrics(cls.cpu(), label.cpu())
                self.update_meters(
                    [meters[i] for i in meters.keys()],
                    [loss, acc, precision, recall, f1])
                _, predicted = torch.max(cls, 1)
                correct = (predicted == label).squeeze()
                if correct.ndim == 0:
                    correct = correct.unsqueeze(0)
                for i in range(label.size(0)):
                    label_i = label[i].item()
                    class_correct[label_i] += correct[i].item()
                    class_total[label_i] += 1
                pbar_test.set_postfix({"loss": loss.item(), "acc": acc})
            # Calculate accuracy for each class
            class_accuracies = self.class_accuracies(class_correct, class_total)
            meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], meters[
                'auc'] = self.calculate_all_metrics(all_preds, all_labels)
            meters['lr'] = self.optimizer.param_groups[0]['lr']
            meters.update(class_accuracies)
            self.print_metrics(meters, prefix=f'Val epoch:{self.epoch}/{self.epochs}')
            if self.args.phase == 'train':
                self.save_checkpoint(meters, 'val')


def main(args, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    from src.nets2 import BCNet
    model = BCNet(num_classes=args.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001, betas=(0.9, 0.99))
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    train_info, val_info = split_pandas(opt.dataset)
    # loss_weight = train_info['label']
    loss_weight = train_info['label'].value_counts().to_dict()
    loss_weight = [1-round(loss_weight[i] / len(train_info), 2) for i in range(len(loss_weight))]
    args.loss_weight = loss_weight
    train_dataset = LungDataset(train_info, opt.dataset, use_ct128=True, use_bbox=True)
    val_dataset = LungDataset(val_info, opt.dataset, use_ct128=True, use_bbox=True, phase='val')
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers
                                    )
    val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers
                                    )
    if args.phase == 'train':
        log_path = makedirs(os.path.join(path, 'logs'))
        model_path = makedirs(os.path.join(path, 'models'))
        args.log_dir = log_path
        args.save_dir = model_path
    trainer = BCTrainer(model=model,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         device=device,
                         train_loader=train_loader,
                         test_loader=val_loader,
                         args=args)

    if args.phase == 'train':
        trainer.train()
    else:
        trainer.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rd', type=int, default=34)
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='./configs/dataset_001.json')

    opt = parser.parse_args()
    args_dict = vars(opt)
    now = time.strftime('%y%m%d%H%M', time.localtime())
    opt.now = now
    path = None
    if opt.phase == 'train':
        path = makedirs(f'./results/{now}')
        with open(os.path.join(path, 'train_config.json'), 'w') as fp:
            json.dump(args_dict, fp, indent=4)
        print(f"Training configuration saved to {now}")
    print(args_dict)
    main(opt, path)