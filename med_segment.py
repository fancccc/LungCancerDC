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
from src.dataloader import split_pandas, my_dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
from src.dataloader import split_data, my_dataloader
from src.resnet import generate_model
import time
import json
from utils import AverageMeter as AverageMeter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.loss import FocalLoss
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from src.med3dnet import generate_model


def load_model(model, checkpoint_path, multi_gpu=False):
    """
    通用加载模型函数。

    :param model: 要加载状态字典的PyTorch模型。
    :param checkpoint_path: 模型权重文件的路径。
    :param multi_gpu: 布尔值，指示是否使用多GPU加载模型。
    :return: 加载了权重的模型。
    """
    # 加载状态字典
    pretrain = torch.load(checkpoint_path)
    if 'model_state_dict' in pretrain.keys():
        state_dict = pretrain['model_state_dict']
    else:
        state_dict = pretrain['state_dict']
    # 检查是否为多卡模型保存的状态字典
    if list(state_dict.keys())[0].startswith('module.'):
        # 移除'module.'前缀（多卡到单卡）
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    for name, param in model.named_parameters():
        if name in state_dict and param.size() == state_dict[name].size():
            param.data.copy_(state_dict[name])
            # print(f"Loaded layer: {name}")
        else:
            print(f"Skipped layer: {name}")
    # 如果需要在多GPU上运行模型
    if multi_gpu:
        # 使用DataParallel封装模型
        model = nn.DataParallel(model)

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
        self.loss_function = FocalLoss(alpha=[1-0.67, 1-0.28, 1-0.025, 1-0.015], device=device)
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

    def calculate_metrics(self, pred, label):
        with torch.no_grad():
            probabilities = torch.softmax(pred, dim=1)
            _, predicted_labels = torch.max(probabilities, 1)
            true_labels = label if not label.is_cuda else label.cpu()
            acc = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='macro')
            recall = recall_score(true_labels, predicted_labels, average='macro')
            f1 = f1_score(true_labels, predicted_labels, average='macro')
            # auc = roc_auc_score(torch.nn.functional.one_hot(label, num_classes=probabilities.shape[1]), probabilities,
            #                     multi_class='ovr', average='macro')

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
            # gtbox = gtbox.to(self.device)
            # clicinal = clicinal.to(self.device)
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
            # 添加这个检查以确保correct始终是一维的
            if correct.ndim == 0:
                correct = correct.unsqueeze(0)
            for i in range(label.size(0)):
                label_i = label[i].item()
                class_correct[label_i] += correct[i].item()
                class_total[label_i] += 1
            pbar_train.set_postfix({"loss": loss.item(), "acc": acc, "class_correct": class_correct, "class_total": class_total})
        # Calculate accuracy for each class
        class_accuracies = {}
        for i in range(4):
            if class_total[i] > 0:
                class_accuracies[i] = 100 * class_correct[i] / class_total[i]
            else:
                class_accuracies[i] = 0  # No samples for this class in the dataset

        print("Train Accuracy per class:", class_accuracies)

        meters['accuracy'], meters['precision'], meters['recall'], meters['f1'], meters['auc'] = self.calculate_all_metrics(all_preds, all_labels)
        self.print_metrics(meters, prefix=f'Epoch: [{self.epoch}/{self.epochs}]')
        self.log_metrics_to_tensorboard(meters, self.epoch, stage_prefix='Train')
        self.log_metrics_to_tensorboard({'lr':self.optimizer.param_groups[0]['lr']}, self.epoch)
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            pbar_test = tqdm(enumerate(self.test_loader))
            for inx, (img, label, gtbox, clicinal, pid) in pbar_test:
                img, label = img.to(self.device), label.to(self.device)
                segs = self.model(img)
                segs = F.softmax(segs, dim=1)
                print(segs.shape)
                _, segs = torch.max(segs, 1)  # 返回最大值和对应的索引，索引即类别标签

                # 遍历批量中的每个图像和对应的分割结果
                for i, seg in enumerate(segs):
                    print(seg.sum(), seg.shape)
                    # 将tensor数据转换为numpy数组
                    seg_np = seg.cpu().numpy()
                    # 转换为SimpleITK图像
                    seg_itk = sitk.GetImageFromArray(seg_np.astype(np.uint8))
                    # 假设你有一个函数或方法来获取原始图像的元数据
                    # 原始图像需要从批量中索引出来
                    original_itk_image = sitk.GetImageFromArray(img.cpu().numpy()[i][0])  # 假设img通道在第0位
                    seg_itk.CopyInformation(original_itk_image)
                    # 保存图像为NIfTI格式
                    sitk.WriteImage(seg_itk, f'./results/output_segmentation_{pid.numpy()[i]}.nii.gz')


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main(args, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    # model = generate_model(model_depth=args.rd, n_classes=args.num_classes)
    model = generate_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
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
    parser.add_argument('--n_seg_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--model_depth', type=int, default=34)
    parser.add_argument('--input_W', type=int, default=256)
    parser.add_argument('--input_H', type=int, default=256)
    parser.add_argument('--input_D', type=int, default=300)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--resnet_shortcut', type=str, default='A')
    parser.add_argument('--no_cuda', type=bool, default=False)


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