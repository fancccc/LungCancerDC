# -*- coding: utf-8 -*-
# Time    : 2023/12/12 20:30
# Author  : fanc
# File    : train.py

# -*- coding: utf-8 -*-
# Time    : 2023/10/30 20:35
# Author  : fanc
# File    : train_base.py

import warnings
warnings.filterwarnings("ignore")
import logging
import os.path
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ExponentialLR
from src.dataloader import split_data, my_dataloader
from torch.nn.parallel import DataParallel
from src.resnet import generate_model
import time
import json
import torch.nn.functional as F
from utils import AverageMeter as AverageMeter
from utils import calculate_accuracy

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
        self.args = args
        self.load_model()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.summaryWriter = summaryWriter

    def __call__(self):
        if self.args.phase == 'train':
            for epoch in tqdm(range(self.epochs)):
                start = time.time()
                self.epoch = epoch+1
                self.train_one_epoch()
                self.num_params = sum([param.nelement() for param in self.model.parameters()])
                self.scheduler.step()
                end = time.time()
                print("Epoch: {}, train time: {}".format(epoch, end - start))
                if epoch % 1 == 0:
                    self.evaluate()
        else:
            self.model.eval()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accuracies = AverageMeter()
            end_time = time.time()
            with torch.no_grad():
                for inx, (x, label) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                    data_time.update(time.time() - end_time)
                    x = x.to(self.device)
                    label = label.to(self.device)
                    out = self.model(x)
                    pred = F.softmax(out, dim=1)
                    print('out:{}, sigmoid out:{}, label:{}'.format(out, pred, label))
                    acc = calculate_accuracy(pred, label)
                    loss = self.loss_function(out, label)
                    losses.update(loss.item(), x.size(0))
                    accuracies.update(acc, x.size(0))
                    batch_time.update(time.time() - end_time)
                    end_time = time.time()
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc {acc.val:.3f} ({acc.avg:.3f})'
                          '\nout:{out}-pred:{pred}-label:{label}'.format(
                            self.epoch,
                            inx + 1,
                            len(self.test_loader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            acc=accuracies,
                            out=out,pred=pred, label=label))

    def load_model(self):
        if self.args.MODEL_WEIGHT:
            pretrain = torch.load(self.args.MODEL_WEIGHT)
            if 'model_state_dict' in pretrain.keys():
                state_dict = 'model_state_dict'
            else:
                state_dict = 'state_dict'
            pre_dict = pretrain[state_dict]
            model_dict = self.model.state_dict()
            for m in model_dict.keys():
                if 'module' in m:
                    pass
                else:
                    old_key = list(pre_dict.keys())
                    for k in old_key:
                        if 'module' in k:
                            pre_dict[k.replace('module.', '')] = pre_dict[k]
                            pre_dict.pop(k)
                break
            # print(model_dict.keys())
            # print(pre_dict.keys())
            pretrained_dict = {k: v for k, v in pre_dict.items() if
                               k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f'pretrain weight can use : {len(pretrained_dict)} / {len(model_dict)}')
            print('load model weight success!')

    def evaluate(self):
        self.model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        end_time = time.time()

        with torch.no_grad():
            for inx, (x, label) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                data_time.update(time.time() - end_time)
                x = x.to(self.device)
                label = label.to(self.device)
                out = self.model(x)
                loss = self.loss_function(out, label)
                pred = F.softmax(out, dim=1)
                acc = calculate_accuracy(pred, label)
                losses.update(loss.item(), x.size(0))
                accuracies.update(acc, x.size(0))
                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    self.epoch,
                    inx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies))

            self.summaryWriter.add_scalars("Loss", {'Test': losses.avg}, self.epoch)
            self.summaryWriter.add_scalars("Acc", {'Test': accuracies.avg}, self.epoch)



            if self.epoch % self.args.save_epoch == 0:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),  # *模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                    'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                    'best_val_score': accuracies.avg,
                    'num_params': self.num_params,
                    'epoch': self.epoch
                }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-%d.pth' % self.epoch))
                logger.logger.info('save model %d successed......\n'%self.epoch)

            if self.best_acc < accuracies.avg:
                self.best_acc = accuracies.avg
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),  # *模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                    'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                    'best_val_score': self.best_acc,
                    'num_params': self.num_params,
                    'epoch': self.epoch
                }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'best_checkpoint.pth'))
                logger.logger.info('save best model successed......\n')

    def train_one_epoch(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        self.model.train()
        end_time = time.time()
        for inx, (x, label) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            data_time.update(time.time() - end_time)
            x = x.to(self.device)
            label = label.to(self.device)
            out = self.model(x)
            loss = self.loss_function(out, label)
            pred = F.softmax(out, dim=1)
            acc = calculate_accuracy(pred, label)
            losses.update(loss.item(), x.size(0))
            accuracies.update(acc, x.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(self.epoch,
                                                             inx + 1,
                                                             len(self.train_loader),
                                                             batch_time=batch_time,
                                                             data_time=data_time,
                                                             loss=losses,
                                                             acc=accuracies))
        self.summaryWriter.add_scalars("Loss", {'Train': losses.avg}, self.epoch)
        self.summaryWriter.add_scalars("Acc", {'Train': accuracies.avg}, self.epoch)
        # self.summaryWriter.add_scalars("lr", self.args.lr, self.epoch)
        self.summaryWriter.add_scalar('Lr', self.optimizer.param_groups[0]['lr'], self.epoch)
        # print(f'train epoch:{self.epoch}/{self.epochs}, Loss:{per_epoch_loss / len(self.train_loader)}, acc:{num_correct / total_step}')

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main(args, logger, path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    model = generate_model(model_depth=args.rd, n_classes=args.num_classes)

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001, betas=(0.9, 0.99))
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    data_dir = args.input_path
    train_infos, val_infos = split_data(data_dir)
    train_loader = my_dataloader(data_dir, train_infos, batch_size=args.batch_size, input_size=args.input_size)
    val_loader = my_dataloader(data_dir, val_infos, batch_size=args.batch_size, input_size=args.input_size)
    summaryWriter = None
    if args.phase == 'train':
        logger.logger.info('start training......\n')
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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rd', type=int, default=50)
    parser.add_argument('--save-epoch', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--input-size', type=str, default="144, 256, 256")
    parser.add_argument('--input-path', type=str, default='/home/wangchangmiao/kidney/data/')
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

    logger = Logger()
    main(opt, logger, path)
