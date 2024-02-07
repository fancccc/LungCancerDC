# -*- coding: utf-8 -*-
# Time    : 2023/12/12 19:33
# Author  : fanc
# File    : dataloader.py

import os
import re

import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import json
from skimage.transform import resize
import SimpleITK as sitk
from scipy.ndimage import zoom
from collections import defaultdict

def split_data(data_dir, info_name, rate=0.8):
    with open(os.path.join(data_dir, info_name), 'r', encoding='utf-8') as f:
        infos = json.load(f)

    # 创建一个字典，用于按类别存储数据
    class_data = defaultdict(list)
    for info in infos:
        label = info['label']
        class_data[label].append(info)

    train_infos = []
    test_infos = []

    # 对每个类别进行分层抽样
    for label, data in class_data.items():
        random.seed(1900)
        random.shuffle(data)
        num_samples = len(data)
        train_num = int(rate * num_samples)
        train_infos.extend(data[:train_num])
        test_infos.extend(data[train_num:])

    return train_infos, test_infos

class MyDataset(Dataset):
    def __init__(self, data_dir, infos, img_dir, phase='train'):

        img_dir = os.path.join(data_dir, img_dir)

        self.img_dir = img_dir
        self.pids = [i['pid'] for i in infos]
        self.phase = phase
        self.labels = torch.tensor([i['label'] for i in infos], dtype=torch.long)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        img = sitk.ReadImage(os.path.join(self.img_dir, f"{self.pids[i]}.nii.gz"))
        if self.phase == 'train':
            img = self.train_preprocess(img)
        else:
            img = self.val_preprocess(img)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = self.labels[i]

        return img, label

    def train_preprocess(self, img):
        img = self.resample(img)
        img = self.normalize(img)
        return img
    def val_preprocess(self, img):
        img = self.resample(img)
        img = self.normalize(img)
        return img

    def resample(self, itkimage):
        img = sitk.GetArrayFromImage(itkimage)
        return np.array(img, dtype=np.float32)

    def normalize(self, img, MIN_BOUND=-1000.0, MAX_BOUND=400.0):

        img[img > MAX_BOUND] = MAX_BOUND
        img[img < MIN_BOUND] = MIN_BOUND
        """数据标准化"""
        img = (img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        return img


def my_dataloader(data_dir, infos, img_dir, batch_size=1, shuffle=True):
    dataset = MyDataset(data_dir, infos, img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':
    data_dir = r'C:\Users\Asus\Desktop\data'
    train_info, test_info = split_data(data_dir, rate=0.8)
    train_dataloader = my_dataloader(data_dir, train_info, input_size=(128, 256, 256), batch_size=1)
    test_dataloader = my_dataloader(data_dir, test_info, input_size=(128, 256, 256), batch_size=1)
    for i, (image, mask, label) in enumerate(train_dataloader):
        print(image, mask.max(), label.shape, label[:, 0].shape)

        new_image = sitk.GetImageFromArray(image.numpy()[0][0])
        new_image.SetSpacing([1, 1, 3])
        sitk.WriteImage(new_image, os.path.join(data_dir, f'{i}.nii.gz'))

        new_mask = sitk.GetImageFromArray(mask.numpy()[0][0])
        new_mask.SetSpacing([1, 1, 3])
        sitk.WriteImage(new_mask, os.path.join(data_dir, f'{i}-mask.nii.gz'))
