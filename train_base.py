# -*- coding: utf-8 -*-
# Time    : 2023/10/30 20:35
# Author  : fanc
# File    : train_base.py

import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from src.dataloader import split_pandas, LungDataset
from torch.utils.data import DataLoader
from src.trainer import BaseTrainer
from tqdm import tqdm
import torch
from utils import makedirs
import time
import json
import numpy as np
from src.loss import FocalLoss
from torch.optim.lr_scheduler import ExponentialLR

class CTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(CTrainer, self).__init__(*args, **kwargs)
        # self.loss_function = FocalLoss(alpha=self.args.loss_weight, device=self.device)
        # self.loss_function = FocalLoss(alpha=[1-0.7, 1-0.1, 1-0.014, 1-0.17], device=self.device)
        # weight = torch.tensor([1 - 0.7, 0.7]).to(self.device)

        # self.loss_function = torch.nn.CrossEntropyLoss(weight=weight)
    def train_one_epoch(self):
        self.model.train()
        meters = self.get_meters()
        all_preds = []
        all_labels = []
        class_correct = list(0. for i in range(self.args.num_classes))
        class_total = list(0. for i in range(self.args.num_classes))
        pbar_train = tqdm(enumerate(self.train_loader))
        for inx, data in pbar_train:
            # data['image'] = data['ct128'].to(self.device)
            # data['clinical'] = data['clinical'].to(self.device)
            # label = data['label'].to(self.device)
            ct, label = data['ct32'].to(self.device), data['label'].to(self.device)
            cls = self.model(ct)
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
        self.his_train_loss.append(meters['loss'].avg.detach().cpu().item())
        self.evaluate()
        # self.plot_curve()
        # else:
        #     self.his_val_loss.append(np.nan)
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
                # data['image'] = data['ct128'].to(self.device)
                # data['clinical'] = data['clinical'].to(self.device)
                # label = data['label'].to(self.device)
                ct, label = data['ct32'].to(self.device), data['label'].to(self.device)
                cls = self.model(ct)
                # print(cls)
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
                'auc'] = self.calculate_all_metrics(all_preds, all_labels, phase='val')
            meters['lr'] = self.optimizer.param_groups[0]['lr']
            meters.update(class_accuracies)
            self.print_metrics(meters, prefix=f'Val epoch:{self.epoch}/{self.epochs}')
            if self.args.phase == 'train':
                self.save_checkpoint(meters, phase='test')
                # self.his_val_loss.append(meters['loss'].avg.cpu().item())


def main(args, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    from src.resnet import generate_model
    model = generate_model(model_depth=args.rd, n_classes=args.num_classes)
    # from multisurv.nets import MultiSurv
    # model = MultiSurv()
    # from comparison.vit import COMP_VIT
    # model = COMP_VIT(img_size=args.image_size, num_classes=args.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001, betas=(0.9, 0.99))
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    train_info, val_info = split_pandas(opt.dataset)

    loss_weight = train_info['label'].apply(lambda x: 2 if x == 3 else x).value_counts().to_dict()
    loss_weight = [1-round(loss_weight[i] / len(train_info), 2) for i in range(len(loss_weight))]
    args.loss_weight = loss_weight
    train_dataset = LungDataset(train_info, opt.dataset, use_ct32=args.use_ct32, use_ct128=args.use_ct128, use_cli=True)
    val_dataset = LungDataset(val_info, opt.dataset, use_ct32=args.use_ct32, use_ct128=args.use_ct128, use_cli=True)
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
    trainer = CTrainer(model=model,
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
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rd', type=int, default=18)
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='./configs/dataset.json')

    opt = parser.parse_args()
    args_dict = vars(opt)
    now = time.strftime('%y%m%d%H%M', time.localtime())
    opt.now = now
    opt.image_size = (32, 32, 32)
    opt.use_ct32 = True
    opt.use_ct128 = False
    opt.net = 'from src.resnet import generate_model'
    path = None
    if opt.phase == 'train':
        # if not os.path.exists(f'./results/{now}'):
        #     os.makedirs(f'./results/{now}')
        path = makedirs(f'./results/{now}')
        opt.path = path
        with open(os.path.join(path, 'train_config.json'), 'w') as fp:
            json.dump(args_dict, fp, indent=4)
        print(f"Training configuration saved to {now}")
    print(args_dict)
    main(opt, path)