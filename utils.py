# -*- coding: utf-8 -*-
# Time    : 2023/12/12 20:32
# Author  : fanc
# File    : utils.py
import torch

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
