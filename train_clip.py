# -*- coding: utf-8 -*-
'''
@file: train_mctrebs.py
@author: fanc
@time: 2024/6/27 21:28
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

class CLIPTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(CLIPTrainer, self).__init__(*args, **kwargs)
        # self.loss_function = FocalLoss(alpha=self.args.loss_weight, device=self.device)
        # weight = torch.tensor([0.64, 0.29]).to(self.device)
        # self.loss_function = torch.nn.CrossEntropyLoss(weight=weight)
        # for name, param in self.model.named_parameters():
        #     param.register_hook(lambda grad, name=name: print(name, grad) if torch.isnan(grad).any() else None)

    def train_one_epoch(self):
        meters = self.get_meters()
        self.model.train()
        all_preds = []
        all_labels = []
        class_correct = list(0. for i in range(self.args.num_classes))
        class_total = list(0. for i in range(self.args.num_classes))
        pbar_train = tqdm(enumerate(self.train_loader))
        for inx, data in pbar_train:
            label = data['label'].to(self.device)
            for k in data.keys():
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(self.device)
            # print(data['bbox32'], data['bbox128'], data['label'])
            cls, loss = self.model(data)
            # assert not torch.isnan(cls).any(), f'data {data}'
            if torch.isnan(loss).any():
                # print(loss)
                self.args.MODEL_WEIGHT = os.path.join(self.args.save_dir, 'model_last.pt')
                self.self_model()
                self.logger.info('loss is NaN, reload thr last weight!')
                return
            #     print(data)
            #     return
            # loss = self.calculate_loss(cls, label)
            # loss = 0.8*loss + 0.2*l
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        # if self.epoch % 5 == 0:
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
                label = data['label'].to(self.device)
                for k in data.keys():
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].to(self.device)
                cls, loss = self.model(data)
                assert not torch.isnan(cls).any(), f'data {data}'
                # loss = self.calculate_loss(cls, label)
                # loss = 0.8 * loss + 0.2 * l
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
            # loss weight auto projection
            # new_loss_weight = class_accuracies.values()
            # new_loss_weight = [(1-i*0.8)*0.8 for i in new_loss_weight]
            # new_loss_weight = [i/sum(new_loss_weight) for i in new_loss_weight]
            # self.model.update_loss_weight(new_loss_weight)
            # # meters['focal loss weight'] = ','.join([str(round(i, 2)) for i in new_loss_weight])
            # print('loss weight : ', ','.join([str(round(i, 2)) for i in new_loss_weight]))

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

    # from CLIP.model import CLIP_VBCRNet5 as MYNET
    # exec(args.netimport + ' as MYNET')
    # from src.tmss import TMSSNet as MYNET

    # base_params = [param for name, param in model.named_parameters() if not name.startswith('CLIP.encode_clinical')]
    # clinical_encoder_module = [param for name, param in model.named_parameters() if name.startswith('CLIP.encode_clinical')]
    # optimizer = torch.optim.Adam([
    #     {'params': base_params, 'lr': 1e-3, 'weight_decay': 1e-4},  # 为基础网络设置较低的学习率
    #     {'params': clinical_encoder_module, 'lr': 1e-4, 'weight_decay': 1e-4}  # 为新添加的分类器设置较高的学习率
    # ])

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    train_info, val_info = split_pandas(args.dataset)
    # loss_weight = train_info['label']
    loss_weight = train_info['label'].value_counts().to_dict()
    loss_weight = [1-round(loss_weight[i] / len(train_info), 2) for i in range(len(loss_weight))]
    args.loss_weight = loss_weight

    from CLIP.clip_tmss import CLIP_TMSS_NetV5 as MYNET
    clinical_length = 27
    if 'CRDC' in args.dataset:
        clinical_length = 8
    model = MYNET(num_classes=args.num_classes, clinical_length=clinical_length, loss_weight=loss_weight, clip_loss_weight=args.clip)
    for name, param in model.named_parameters():
        if 'classifier' not in name:  # 只保留全连接层可训练
            param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.001, betas=(0.9, 0.99))
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    train_dataset = LungDataset(train_info, args.dataset, use_ct32=True, use_ct128=True, use_radiomics=False, use_cli=True, use_bbox=True, use_seg=False)
    val_dataset = LungDataset(val_info, args.dataset, phase='val', use_ct32=True, use_ct128=True, use_radiomics=False, use_cli=True, use_bbox=True, use_seg=False)
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
    trainer = CLIPTrainer(model=model,
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
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='./configs/dataset.json')

    opt = parser.parse_args()

    now = time.strftime('%y%m%d%H%M', time.localtime())
    opt.now = now
    with open(opt.dataset, 'r')as f:
        config = json.load(f)
    path = None
    # opt.netimport = config['net']['netimport']
    opt.net = 'from CLIP.clip_tmss import CLIP_TMSS_NetV5 as MYNET'
    if opt.phase == 'train':
        path = makedirs(f'./results/{now}')
        args_dict = vars(opt)
        args_dict.update(config)
        with open(os.path.join(path, 'train_config.json'), 'w') as fp:
            json.dump(args_dict, fp, indent=4)
        print(f"Training configuration saved to {now}")
        print(args_dict)
    main(opt, path)