# -*- coding: utf-8 -*-
'''
@file: train_old.py
@author: fanc
@time: 2024/5/17 11:49
'''
import warnings
warnings.filterwarnings("ignore")
import os.path
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import time
from utils import AverageMeter as AverageMeter
from utils import load_model, setup_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
class BaseTrainer:
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
        self.val_best_acc = 0
        self.val_best_acc_epoch = 0
        self.args = args
        self.self_model()
        self.loss_function = nn.CrossEntropyLoss()
        self.num_params = 0
        self.his_train_metrics = []
        self.his_train_loss = []
        self.his_val_metrics = []
        self.his_val_loss = []
    def __call__(self):
        pass

    def train(self):
        self.logger = setup_logger(self.args.log_dir, name=self.args.now)
        self.logger.info("Trainer has been set up.")
        for epoch in tqdm(range(self.epochs)):
            start = time.time()
            self.epoch = epoch + 1
            self.train_one_epoch()
            self.num_params = sum([param.nelement() for param in self.model.parameters()])
            self.scheduler.step()
            end = time.time()
            self.logger.info("Epoch: {}, train time: {}, params: {}".format(epoch, end - start, self.num_params))

    def self_model(self):
        if self.args.MODEL_WEIGHT:
            self.load_model(self.args.MODEL_WEIGHT)
            # self.model = load_model(model=self.model,
            #                 checkpoint_path=self.args.MODEL_WEIGHT,
            #                 multi_gpu=torch.cuda.device_count() > 1)
            # print('load model weight success!')
        self.model = self.model.to(self.device)
        self.num_params = sum([param.nelement() for param in self.model.parameters()])
        print('num_params: {}'.format(self.num_params))

    def load_model(self, checkpoint_path, multi_gpu=False):
        print("Loading model...")
        data = torch.load(checkpoint_path)
        if 'model_state_dict' in data.keys():
            state_dict = data['model_state_dict']
        else:
            state_dict = data['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        if multi_gpu:
            self.model = nn.DataParallel(self.model)
        self.epoch = data['epoch']
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.scheduler.load_state_dict(data['scheduler_state_dict'])
        print("Finished loading model!")

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
    def calculate_all_metrics(self, pred, label, phase='train'):
        pred = torch.tensor(pred)
        label = torch.tensor(label)
        # print(pred.shape, label.shape)
        probabilities = torch.softmax(pred, dim=1)
        _, predicted_labels = torch.max(probabilities, 1)
        if self.args.phase == 'val':
            res = {}
            res['true_label'] = label.numpy()
            res['predicted'] = probabilities.numpy()
            # import pandas as pd
            # res = pd.DataFrame([_.numpy(), label.numpy()]).T
            save_dir = os.path.dirname(self.args.MODEL_WEIGHT)
            np.save(os.path.join(save_dir, 'pred.npy'), res)
            # with open(os.path.join(save_dir, 'pred.json'), 'w') as f:
            #     json.dump(res, f)


        label = label.numpy()
        predicted_labels = predicted_labels.numpy()

        acc = accuracy_score(label, predicted_labels)
        precision = precision_score(label, predicted_labels, average='macro')
        recall = recall_score(label, predicted_labels, average='macro')
        f1 = f1_score(label, predicted_labels, average='macro')
        # ck = cohen_kappa_score(label, predicted_labels)
        # print(ck)
        # print('weight', recall_score(label, predicted_labels, average='weighted'))
        try:
            one_hot_labels = torch.nn.functional.one_hot(torch.tensor(label), num_classes=probabilities.shape[1])
            auc = roc_auc_score(one_hot_labels, probabilities.numpy(),
                                multi_class='ovr', average='macro')
        except Exception as e:
            print(e)
            print(f'label unique: {np.unique(label)}')
            print(f'probabilities shape: {probabilities.shape}')
            print(f'probabilities: {probabilities}')
            auc = 0
        if phase == 'train':
            self.his_train_metrics.append([acc, precision, recall, f1])
            self.his_val_metrics.append([np.nan, np.nan, np.nan, np.nan])
        else:
            self.his_val_metrics.append([acc, precision, recall, f1])
        return acc, precision, recall, f1, auc

    def plot_curve(self):
        if len(self.his_val_loss) > 0:
            self.plot_loss_curve(self.his_train_loss, self.his_val_loss)
        if len(self.his_val_metrics) > 0:
            self.plot_accuracy_curve([i[0] for i in self.his_train_metrics],
                                     [i[0] for i in self.his_val_metrics])
    def plot_loss_curve(self, train_losses, val_losses):
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.savefig(os.path.join(self.args.log_dir, f'loss_curve.png'))

    def plot_accuracy_curve(self, train_accs, val_accs):
        plt.figure()
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy Curve')
        plt.legend()
        plt.savefig(os.path.join(self.args.log_dir, f'acc_curve.png'))
    def calculate_loss(self, pred, label):
        # criterion = nn.CrossEntropyLoss()
        loss = self.loss_function(pred, label)
        return loss

    def class_accuracies(self, class_correct, class_total):
        class_accuracies = {}
        for i in range(self.args.num_classes):
            if class_total[i] > 0:
                class_accuracies[i] = class_correct[i] / class_total[i]
            else:
                class_accuracies[i] = 0  # No samples for this class in the dataset
        return class_accuracies

    def get_meters(self):
        meters = {
            'loss': AverageMeter(),
            'accuracy': AverageMeter(), 'precision': AverageMeter(), 'recall': AverageMeter(),'f1': AverageMeter()
        }
        return meters

    def update_meters(self, meters, values):
        for meter, value in zip(meters, values):
            meter.update(value)

    def print_metrics(self, meters, prefix=""):
        for meter, value in meters.items():
            try:
                self.logger.info(f'{prefix} {meter}: {value.avg:.4f}' if isinstance(value, AverageMeter) else f'{prefix} {meter}: {value:.4f}')
            except:
                print(f'{prefix} {meter}: {value.avg:.4f}' if isinstance(value, AverageMeter) else f'{prefix} {meter}: {value:.4f}')
    def train_one_epoch(self):
        pass
    def save_checkpoint(self, meters, phase='train'):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'meters': {k: v for k, v in meters.items() if not isinstance(v, AverageMeter)},
            'num_params': self.num_params
        }
        torch.save(checkpoint, os.path.join(self.args.save_dir, 'model_last.pt'))
        self.logger.info(f"New checkpoint saved at epoch {self.epoch} with accuracy: {meters['accuracy']:.4f}")
        self.logger.info(f"best checkpoint saved at epoch {self.val_best_acc_epoch} with accuracy: {self.val_best_acc:.4f}")
        if phase == 'train':
            if meters['accuracy'] > self.best_acc:
                self.best_acc_epoch = self.epoch
                self.best_acc = meters['accuracy']
                torch.save(checkpoint, os.path.join(self.args.save_dir, f'model_best_{phase}.pt'))
                self.logger.info(f"New best model saved at epoch {self.epoch} with train accuracy: {self.best_acc:.4f}")
        else:
            if meters['accuracy'] > self.val_best_acc:
                self.val_best_acc_epoch = self.epoch
                self.val_best_acc = meters['accuracy']
                torch.save(checkpoint, os.path.join(self.args.save_dir, f'model_best_{phase}.pt'))
                self.logger.info(f"New best model saved at epoch {self.epoch} with val accuracy: {self.val_best_acc:.4f}")
    def evaluate(self):
        pass