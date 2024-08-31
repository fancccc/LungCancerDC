# -*- coding: utf-8 -*-
# Time    : 2023/12/12 19:33
# Author  : fanc
# File    : dataloader.py

import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import json
import SimpleITK as sitk
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
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

def get_conf(file):
    with open(file, 'r', encoding='utf-8') as f:
        conf = json.load(f)
    return conf

def split_pandas(file):
    seeds = 1900
    conf = get_conf(file)
    df = pd.read_csv(os.path.join(conf['data_dir'], conf['infos_name']), encoding='gb18030')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seeds, stratify=df['label'])
    return train_df, test_df

def sliding_window(image, window_size, step_size):
    # 对每个维度计算滑动窗口的起止点
    for z in range(0, max(image.shape[0] - window_size[0], 0) + 1, step_size):
        for y in range(0, max(image.shape[1] - window_size[1], 0) + 1, step_size):
            for x in range(0, max(image.shape[2] - window_size[2], 0) + 1, step_size):
                # 确保窗口不超出图像边界
                end_z = min(z + window_size[0], image.shape[0])
                end_y = min(y + window_size[1], image.shape[1])
                end_x = min(x + window_size[2], image.shape[2])
                # 提取窗口区域
                window = image[z:end_z, y:end_y, x:end_x]
                # # 如果窗口大小不足，进行填充（可选）
                # if window.shape != tuple(window_size):
                #     padding = [(0, window_size[i] - window.shape[i]) for i in range(3)]
                #     window = np.pad(window, padding, mode='constant', constant_values=0)
                yield window
class MyDataset(Dataset):
    def __init__(self, df, conf, crop_size=128, phase='train', ni=True):
        self.conf = conf
        self.df = df.reset_index(drop=True)
        data_dir = self.conf['data_dir']
        self.img_dir = os.path.join(data_dir, self.conf['img_dir'])
        self.mask_dir = os.path.join(data_dir, self.conf['mask_dir'])
        self.pids = df['pid'].tolist()
        self.phase = phase
        self.labels = torch.tensor(df['label'].tolist(), dtype=torch.long)
        self.crop_size = crop_size
        self.ni = ni
        # self.indices = torch.tensor([2, 1, 0, 5, 4, 3], dtype=torch.long)
    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        if self.ni:
            img = sitk.ReadImage(os.path.join(self.img_dir, f"{self.pids[i]}.nii.gz"))
            mask = sitk.ReadImage(os.path.join(self.mask_dir, f"lcdc_{self.pids[i].replace('P', '')}.nii.gz"))
            bbox = eval(self.df.iloc[i]['bbox'])
        else:
            img, mask, bbox = None, None, None
        # x, y, z, d, h, w  to z, y, x, w, h, d
        # bounding_box = torch.tensor(bounding_box)[self.indices]
        keys = torch.tensor(range(0, 47), dtype=torch.long)
        values = torch.tensor(self.df.iloc[i].drop(labels=['pid', 'c', 'label', 'bbox', 'bbox_num']).values.tolist(), dtype=torch.float)
        clicinal = torch.stack((keys, values))
        label = self.labels[i]
        # print("Bounding Box:", bounding_box, 'clicinal:', clicinal.shape)
        if self.phase == 'train':
            img, mask, bbox = self.train_preprocess(img, mask, bbox)
            if self.ni:
                return img, mask, bbox, label, clicinal, self.pids[i]
            else:
                return label, clicinal, self.pids[i]
        else:
            patches, bbox = self.val_preprocess(img, mask, bbox)
            if self.ni:
                return patches, bbox, label, clicinal, self.pids[i]
            else:
                return label, clicinal, self.pids[i]

    def cropping(self, img, mask, bounding_box):
        x, y, z, w, h, d = bounding_box
        crop_size = self.crop_size
        # center_x = x + w // 2
        # center_y = y + h // 2
        # center_z = z + d // 2
        center_x = x
        center_y = y
        center_z = z
        half_size = crop_size // 2
        def compute_start(center, min_val, max_val):
            start = center - half_size
            offset = np.random.randint(-half_size // 1, half_size // 1)
            start += offset
            # 确保裁剪区域不超出图像边界
            start = max(min_val, min(start, max_val - crop_size))
            return int(start)

        start_x = compute_start(center_x, 0, img.shape[0])
        start_y = compute_start(center_y, 0, img.shape[1])
        start_z = compute_start(center_z, 0, img.shape[2])
        # print(start_x)

        cropped_img = img[start_x:start_x + crop_size, start_y:start_y + crop_size, start_z:start_z + crop_size]
        cropped_mask = mask[start_x:start_x + crop_size, start_y:start_y + crop_size, start_z:start_z + crop_size]

        new_x_min = max(0, x - start_x)
        new_y_min = max(0, y - start_y)
        new_z_min = max(0, z - start_z)
        new_width = min(crop_size, w - max(0, start_x - (x - w // 2)))
        new_height = min(crop_size, h - max(0, start_y - (y - h // 2)))
        new_depth = min(crop_size, d - max(0, start_z - (z - d // 2)))
        new_bbox = [new_x_min, new_y_min, new_z_min, new_width, new_height, new_depth]

        return cropped_img, cropped_mask, new_bbox
    def train_preprocess(self, img, mask, bbox):
        if self.ni:
            img = self.get_array(img)
            mask = self.get_array(mask)
            img, mask, bbox = self.cropping(img, mask, bbox)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
            bbox = torch.tensor(bbox, dtype=torch.float32)
        return img, mask, bbox
    def val_preprocess(self, img, mask, bbox):
        # seg
        if self.ni:
            img = self.get_array(img)
            mask = self.get_array(mask)
            # img = np.multiply(img, mask)
            patch_size = (128, 128, 128)
            step_size = 32
            patches = [torch.tensor(window, dtype=torch.float32).unsqueeze(0) for window in sliding_window(img, patch_size, step_size)]
            bbox = torch.tensor(bbox, dtype=torch.float32)
        else:
            patches = img
        return patches, bbox

    def get_array(self, itkimage):
        img = sitk.GetArrayFromImage(itkimage)
        img = img.transpose(2, 1, 0)
        return np.array(img, dtype=np.float32)

    def normalize(self, img, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
        img[img > MAX_BOUND] = MAX_BOUND
        img[img < MIN_BOUND] = MIN_BOUND
        """数据标准化"""
        img = (img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        return img

class LungSliceDataset(Dataset):
    def __init__(self, df, phase='train'):
        self.conf = get_conf()
        self.df = df.reset_index(drop=True)
        data_dir = self.conf['data_dir']
        self.img_dir = os.path.join(data_dir, self.conf['slice_dir'])
        # self.mask_dir = os.path.join(data_dir, self.conf['mask_dir'])
        self.pids = df['pid'].tolist()
        self.phase = phase
        self.labels = torch.tensor(df['label'].tolist(), dtype=torch.long)
        self.ts = transforms.ToTensor()
        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose([
            ])
    def __len__(self):
        return len(self.pids)
    def __getitem__(self, i):
        img_path = os.path.join(self.img_dir, f'{self.pids[i]}.png')
        img = Image.open(img_path).convert('RGB')
        label = self.labels[i]
        # print("Bounding Box:", bounding_box, 'clicinal:', clicinal.shape)
        if self.phase == 'train':
            if label == 3 or label == 2:
                img = self.transform(img)
        img = self.ts(img)
        label = torch.tensor(label, dtype=torch.long)
        pid = self.pids[i]
        return img, label, pid

class LungDataset(Dataset):
    def __init__(self,
                 df, file, phase='train',
                 use_cli=False,
                 use_ct32=False,
                 use_ct64=False,
                 use_ct128=False,
                 use_ct256=False,
                 use_bbox=False,
                 use_mask=False,
                 use_seg=False,
                 use_slice=False,
                 use_radiomics=False):
        self.conf = get_conf(file)
        self.df = df.reset_index(drop=True)
        self.data_dir = self.conf['data_dir']
        self.bids = df['bid'].tolist()
        self.phase = phase
        self.labels = torch.tensor(df['label'].apply(lambda x: 2 if x == 3 else x).tolist(), dtype=torch.long)
        self.use_cli = use_cli
        self.use_bbox = use_bbox
        self.use_mask = use_mask
        self.use_seg = use_seg
        self.use_ct32 = use_ct32
        self.use_ct64 = use_ct64
        self.use_ct128 = use_ct128
        self.use_ct256 = use_ct256
        self.use_slice = use_slice
        self.use_radiomics = use_radiomics
        # if self.use_slice:
        #     self.ts = transforms.ToTensor()
        if self.use_cli:
            cols = list(filter(lambda x: x.startswith('f'),  df.columns.tolist()))
            self.clinical = df[cols].fillna(0)
        #     try:
        #         self.clinical = pd.read_csv(os.path.join(self.data_dir, self.conf['clinical_dir']), encoding='gb18030')
        #     except:
        #         self.clinical = pd.read_csv(os.path.join(self.data_dir, self.conf['clinical_dir']))
        #     self.clinical = pd.merge(self.df[['pid']], self.clinical, how='left', on=['pid'])
        if self.use_radiomics:
            self.radiomics = pd.read_csv(os.path.join(self.data_dir, self.conf['radiomics_dir']))
            self.radiomics = pd.merge(self.df[['bid']], self.radiomics, how='left', on=['bid'])
            self.radiomics = self.radiomics.fillna(0)#method='pad')
            #
            # radiomics_features = self.radiomics.drop('bid', axis=1)
            # from sklearn.preprocessing import StandardScaler
            # scaler = StandardScaler()
            # scaled_features = scaler.fit_transform(radiomics_features)
            # scaled_features_df = pd.DataFrame(scaled_features, columns=radiomics_features.columns)
            # self.radiomics = pd.concat([self.radiomics[['uid']], scaled_features_df], axis=1)

    def __len__(self):
        return len(self.bids)
    def __getitem__(self, i):
        # ct, mask, clinical, bbox = torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0,))
        ct32, mask, clinical, bbox, slice, ct64, ct128, ct256, seg, radiomic = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        bbox32, bbox128 = 0, 0
        bid = self.bids[i]
        label = self.labels[i]
        if self.use_ct32:
            ct32 = self.get_ct_32(i)
        if self.use_ct64:
            ct64 = self.get_ct_64(i)
        if self.use_ct128:
            ct128 = self.get_ct_128(i)
        if self.use_ct256:
            ct256 = self.get_ct_256(i)
        if self.use_bbox:
            bbox32 = self.get_bbox(i, 'bbox32')
            bbox128 = self.get_bbox(i, 'bbox128')
        if self.use_mask:
            pass
        if self.use_cli:
            clinical = self.get_clinical(i)
        if self.use_slice:
            slice = self.get_slice(i)
        if self.use_radiomics:
            radiomic = self.get_radiomic(i)
        res = {'label': label, 'bid': bid, 'ct32': ct32, 'ct64': ct64, 'ct128': ct128,'ct256': ct256,
            'clinical': clinical, 'bbox': bbox, 'mask': mask, 'slice': slice, 'seg': seg, 'radiomics': radiomic,
               'bbox32': bbox32, 'bbox128': bbox128}
        return res

    def get_ct_32(self, i):
        ct_path = os.path.join(self.data_dir, self.conf['img_dir_32'], f'{self.bids[i]}.nii.gz')
        # self.get_nii_file(ct_path)
        return self.get_nii_file(ct_path)
    def get_ct_64(self, i):
        ct_path = os.path.join(self.data_dir, self.conf['img_dir_64'], f'{self.bids[i]}.nii.gz')
        # self.get_nii_file(ct_path)
        return self.get_nii_file(ct_path)
    def get_ct_128(self, i):
        # seg = 0
        ct_path = os.path.join(self.data_dir, self.conf['img_dir_128'], f'{self.bids[i]}.nii.gz')
        if not os.path.exists(ct_path):
            ct_path = os.path.join(self.data_dir, self.conf['img_dir_128'], f'{self.bids[i]}.npy')
        # if self.use_seg:
        #     seg_path = os.path.join(os.path.dirname(ct_path), 'seg', os.path.basename(ct_path))
        #     seg = self.get_nii_file(seg_path, normalize=False)
        img = self.get_nii_file(ct_path)
        return img

    def get_ct_256(self, i):
        ct_path = os.path.join(self.data_dir, self.conf['img_dir_256'], f'{self.uids[i]}.nii.gz')
        if not os.path.exists(ct_path):
            ct_path = os.path.join(self.data_dir, self.conf['img_dir_256'], f'{self.uids[i]}.npy')
        self.get_nii_file(ct_path)
        return self.get_nii_file(ct_path)
    def get_clinical(self, i):
        # values = torch.tensor(self.df.iloc[i].drop(labels=['pid', 'c', 'label', 'bbox', 'bbox_num']).values.tolist(), dtype=torch.float)
        # values = torch.tensor(self.clinical.iloc[i].drop(labels=['pid']).values.tolist(), dtype=torch.float)
        # keys = torch.tensor(range(0, values.shape[0]), dtype=torch.long)
        # clicinal = torch.stack((keys, values))
        values = torch.tensor(self.clinical.iloc[i].values.tolist(), dtype=torch.float)
        return values#.unsqueeze(-1)
    def get_bbox(self, i, btype='bbox'):
        bbox = eval(self.df.iloc[i][btype])
        if '32' in btype:
            img_size = [32, 32, 32]
        elif '128' in btype:
            img_size = [128, 128, 32]
        # img_size = eval(self.df.iloc[i]['img_size'])
        bbox[-1] = max(bbox[-3:])
        bbox[0] /= img_size[0]
        bbox[1] /= img_size[1]
        bbox[2] /= img_size[2]
        bbox[3] /= img_size[0]
        bbox[4] /= img_size[1]
        bbox[5] /= img_size[2]
        bbox = torch.tensor(bbox)
        return bbox
    def get_slice(self, i):
        slice_path = os.path.join(self.data_dir, self.conf['slice_dir'], f'{self.pids[i]}.png')
        slice = Image.open(slice_path).convert('RGB')
        slice = self.ts(slice)
        return slice

    def get_radiomic(self, i):
        values = torch.tensor(self.radiomics.iloc[i].drop(labels=['bid']).values.tolist(), dtype=torch.float)
        return values#.unsqueeze(-1)

    def get_nii_file(self, file, normalize=True):
        if file.endswith('.nii.gz'):
            img = sitk.ReadImage(file)
            img = sitk.GetArrayFromImage(img).transpose(1, 2, 0)
        elif file.endswith('.npy'):
            img = np.load(file)
        if normalize:
            img = self.normalize(img)
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    def normalize(self, img, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
        img[img > MAX_BOUND] = MAX_BOUND
        img[img < MIN_BOUND] = MIN_BOUND
        """数据标准化"""
        img = (img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        return img



def my_dataloader(infos, batch_size=1, shuffle=True, phase='train', crop_size=128, ni=True):
    dataset = MyDataset(infos, phase=phase, crop_size=crop_size, ni=ni)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

class ThreeDSlicesDataset(Dataset):
    def __init__(self, image, gtbox, label):
        self.image = image
        self.gtbox = gtbox
        self.cls_label = label

    def __len__(self):
        return self.image.size(0)

    def __getitem__(self, index):
        img = self.image[index]
        exis_label = 0 if index+1 < self.gtbox[0] - self.gtbox[3]//2 or index+1 > self.gtbox[0] + self.gtbox[3]//2 else 1
        return img.unsqueeze(0), exis_label, self.cls_label

if __name__ == '__main__':
    train_info, val_info = split_pandas('../configs/dataset.json')
    train_dataset = LungDataset(train_info, '../configs/dataset.json', use_ct32=True, use_ct128=True, use_radiomics=True, use_cli=True, use_bbox=True)
    val_dataset = LungDataset(val_info, '../configs/dataset.json', phase='val', use_ct32=True, use_ct128=True, use_radiomics=True, use_cli=True, use_bbox=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=2,
                                shuffle=True,
                                num_workers=6
                                    )
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=6
                                    )
    for res in train_loader:
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    print(k, v)
                    break
        # pass

    # data_dir = '/home/zcd/codes/LungCancerDC/results/dataloader'
    # train_info, val_info = split_pandas()
    # train_loader = my_dataloader(train_info,
    #                             batch_size=1,
    #                             shuffle=True,
    #                                 )
    # val_loader = my_dataloader(val_info,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             phase='test'
    #                                 )
    # for inx, (img, mask, gtbox, label, clicinal, pid) in enumerate(train_loader):
    #     print(torch.multiply(img, mask).shape)
    #     # pid = pid[0]
    #     # new_image = sitk.GetImageFromArray(np.multiply(img.numpy()[0][0], mask.numpy()[0][0].astype(np.uint8)).transpose(2, 1, 0))
    #     # sitk.WriteImage(new_image, os.path.join(data_dir, f'{pid}.nii.gz'))
    #     # # new_mask = sitk.GetImageFromArray(mask.numpy()[0][0].astype(np.uint8).transpose(2, 1, 0))
    #     # # sitk.WriteImage(new_mask, os.path.join(data_dir, f'{pid}-mask.nii.gz'))
    #     # new_bbox = np.zeros_like(img.numpy()[0][0])
    #     # gt_bbox = gtbox.numpy()[0]
    #     # # print(gt_bbox)
    #     # new_bbox[int(gt_bbox[0]-gt_bbox[3]//2):int(gt_bbox[0]+gt_bbox[3]//2),
    #     #          int(gt_bbox[1]-gt_bbox[4]//2):int(gt_bbox[1]+gt_bbox[4]//2),
    #     #          int(gt_bbox[2]-gt_bbox[5]//2):int(gt_bbox[2]+gt_bbox[5]//2)] = 1
    #     # print(f'{pid} bbox sum：{new_bbox.sum()}, gt_bbox: {gt_bbox}')
    #     # new_bbox = sitk.GetImageFromArray(new_bbox.transpose(2, 1, 0))
    #     # sitk.WriteImage(new_bbox, os.path.join(data_dir, f'{pid}-bbox.nii.gz'))
    #
    #     if inx >= 3:
    #         break

