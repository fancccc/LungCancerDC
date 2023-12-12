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

def split_data(data_dir, rate=0.8):
    with open(os.path.join(data_dir, 'infos.json'), 'r', encoding='utf-8') as f:
        infos = json.load(f)

    # 创建一个字典，用于按类别存储数据
    class_data = defaultdict(list)
    for info in infos:
        label = info['c']
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
    def __init__(self, data_dir, infos, input_size, phase='train'):
        '''
        task: 0 :seg,  1 :cla
        '''
        task = [int(i) for i in re.findall('\d', str(task))]
        img_dir = os.path.join(data_dir, 'imgs_nii')

        self.input_size = tuple([int(i) for i in re.findall('\d+', str(input_size))])
        self.img_dir = img_dir
        self.labels = [i['c'] for i in infos]
        self.pids = [i['pid'] for i in infos]
        self.phase = phase
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        img = sitk.ReadImage(os.path.join(self.img_dir, f"{self.ids[i]}.nii.gz"))
        if self.phase == 'train':
            img, mask = self.train_preprocess(img)
        else:
            img, mask = self.val_preprocess(img)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = self.labels[i].unsqueeze(0)

        return img, label

    def train_preprocess(self, img):
        img = self.resample(img)
        # mask = self.resample(mask)
        # print(img.shape, mask.shape)
        img = self.normalize(img)
        img = self.resize(img)
        return img
    def val_preprocess(self, img):
        img = self.resample(img)
        img = self.normalize(img)
        img = self.resize(img)

        return img

    def crop(self, img, mask):
        crop_img = img
        crop_mask = mask
        # amos kidney mask
        crop_mask[crop_mask == 2] = 1
        crop_mask[crop_mask != 1] = 0
        target = np.where(crop_mask == 1)
        [d, h, w] = crop_img.shape
        [max_d, max_h, max_w] = np.max(np.array(target), axis=1)
        [min_d, min_h, min_w] = np.min(np.array(target), axis=1)
        [target_d, target_h, target_w] = np.array([max_d, max_h, max_w]) - np.array([min_d, min_h, min_w])
        z_min = int((min_d - target_d / 2) * random.random())
        y_min = int((min_h - target_h / 2) * random.random())
        x_min = int((min_w - target_w / 2) * random.random())

        z_max = int(d - ((d - (max_d + target_d / 2)) * random.random()))
        y_max = int(h - ((h - (max_h + target_h / 2)) * random.random()))
        x_max = int(w - ((w - (max_w + target_w / 2)) * random.random()))

        z_min = np.max([0, z_min])
        y_min = np.max([0, y_min])
        x_min = np.max([0, x_min])

        z_max = np.min([d, z_max])
        y_max = np.min([h, y_max])
        x_max = np.min([w, x_max])

        z_min = int(z_min)
        y_min = int(y_min)
        x_min = int(x_min)

        z_max = int(z_max)
        y_max = int(y_max)
        x_max = int(x_max)
        crop_img = crop_img[z_min: z_max, y_min: y_max, x_min: x_max]
        crop_mask = crop_mask[z_min: z_max, y_min: y_max, x_min: x_max]

        return crop_img, crop_mask

    def resample(self, itkimage, new_spacing=[1, 1, 1]):
        # spacing = itkimage.GetSpacing()
        img = sitk.GetArrayFromImage(itkimage)
        # # MASK 膨胀腐蚀操作
        # kernel = ball(5)  # 3D球形核
        # # 应用3D膨胀
        # dilated_mask = dilation(mask, kernel)
        # mask = closing(dilated_mask, kernel)
        # resize_factor = spacing / np.array(new_spacing)
        # resample_img = zoom(img, resize_factor, order=0)
        # resample_mask = zoom(mask, resize_factor, order=0, mode='nearest')
        return np.array(img, dtype=np.float32)

    def normalize(self, img, MIN_BOUND=-1000.0, MAX_BOUND=400.0):

        img[img > MAX_BOUND] = MAX_BOUND
        img[img < MIN_BOUND] = MIN_BOUND
        """数据标准化"""
        img = (img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img[img > 1] = 1.
        img[img < 0] = 0.
        return img

    def resize(self, img):
        # img = np.transpose(img, (2, 1, 0))
        # mask = np.transpose(mask, (2, 1, 0))
        rate = np.array(self.input_size) / np.array(img.shape)
        img = zoom(img, rate.tolist(), order=0)
        return img



def my_dataloader(data_dir, infos, batch_size=1, shuffle=True, num_workers=0, input_size=(64, 128, 256)):
    dataset = MyDataset(data_dir, infos, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
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
