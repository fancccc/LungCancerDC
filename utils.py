# -*- coding: utf-8 -*-
# Time    : 2023/12/12 20:32
# Author  : fanc
# File    : utils.py
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import logging
import os
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = 100 * correct / total
    return accuracy

def calculate_acc_sigmoid(outputs, targets):
    batch_size = targets.size(0)
    pred = torch.round(outputs)
    correct = (pred == targets).float()
    n_correct_elems = correct.sum().item()
    return n_correct_elems / batch_size


def evaluate_metrics(prob, preds, labels):
    # 转换为numpy数组，如果使用Tensor
    prob = prob.cpu().numpy() if isinstance(prob, torch.Tensor) else prob
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    overall_accuracy = accuracy_score(labels, preds)

    cm = confusion_matrix(labels, preds)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    recall = recall_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    auc = roc_auc_score(labels, prob, multi_class='ovr')

    return overall_accuracy, class_accuracy, recall, precision, f1, auc

def load_model(model, checkpoint_path, multi_gpu=False):
    print("Loading model...")
    pretrain = torch.load(checkpoint_path)
    if 'model_state_dict' in pretrain.keys():
        state_dict = pretrain['model_state_dict']
    else:
        state_dict = pretrain['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    # unmatched_pretrained = set(state_dict.keys())
    # unmatched_model = set(param for param, _ in model.named_parameters())
    # for name, param in model.named_parameters():
    #     if name in state_dict and param.size() == state_dict[name].size():
    #         param.data.copy_(state_dict[name])
    #         unmatched_pretrained.remove(name)
    #         unmatched_model.remove(name)
    #     else:
    #         print(f"Skipped layer: {name} due to mismatch.")
    # # for name, param in model.named_parameters():
    # #     if name in state_dict and param.size() == state_dict[name].size():
    # #         param.data.copy_(state_dict[name])
    # #         # print(f"Loaded layer: {name}")
    # if unmatched_pretrained:
    #     print(f"Unmatched pretrained layers: {unmatched_pretrained}")
    # if unmatched_model:
    #     print(f"Unmatched model layers: {unmatched_model}")
    if multi_gpu:
        model = nn.DataParallel(model)
    print("Finished loading model!")
    return model

def setup_logger(log_directory, name='filename', log_filename='training.log'):
    # 确保日志目录存在，如果不存在则创建
    # if not os.path.exists(log_directory):
    #     os.makedirs(log_directory)

    # 创建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置日志记录的最低级别

    # 创建一个handler，用于写入日志文件，指定路径
    file_path = os.path.join(log_directory, log_filename)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)

    # 创建一个handler，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def makedirs(path):
    os.makedirs(path, exist_ok=True)
    return path