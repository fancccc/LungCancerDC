import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=None, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, device=device)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()

class Box3DLoss(nn.Module):
    def __init__(self):
        super(Box3DLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, preds, targets):
        """
        计算3D边界框回归损失
        参数:
            preds: 预测的3D边界框，形状为(batch_size, 6)，其中6对应于(x, y, z, h, w, l)
            targets: 真实的3D边界框，形状与preds相同
        返回:
            loss: 3D边界框回归损失
        """
        # 计算Smooth L1损失
        loss = self.smooth_l1_loss(preds, targets)
        return loss

def iou_loss_3d(preds, targets):
    """
    计算3D IoU损失
    参数:
        preds: 预测的3D边界框，形状为(batch_size, 6)，其中6对应于(x, y, z, x_size, y_size, z_size)
        targets: 真实的3D边界框，形状与preds相同
    返回:
        loss: IoU损失
    """
    # 计算交集的体积
    inter_xmin = torch.max(preds[:, 0], targets[:, 0])
    inter_ymin = torch.max(preds[:, 1], targets[:, 1])
    inter_zmin = torch.max(preds[:, 2], targets[:, 2])
    inter_xmax = torch.min(preds[:, 0] + preds[:, 3], targets[:, 0] + targets[:, 3])
    inter_ymax = torch.min(preds[:, 1] + preds[:, 4], targets[:, 1] + targets[:, 4])
    inter_zmax = torch.min(preds[:, 2] + preds[:, 5], targets[:, 2] + targets[:, 5])
    inter_volume = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0) * torch.clamp(inter_zmax - inter_zmin, min=0)

    # 计算预测框和真实框的体积
    preds_volume = preds[:, 3] * preds[:, 4] * preds[:, 5]
    targets_volume = targets[:, 3] * targets[:, 4] * targets[:, 5]

    # 计算并集的体积
    union_volume = preds_volume + targets_volume - inter_volume

    # 计算IoU
    iou = inter_volume / torch.clamp(union_volume, min=1e-6)

    # 计算IoU损失
    loss = 1 - iou
    return loss.mean()


class FocalLoss3dMap(nn.Module):
    def __init__(self, alpha=2, beta=4, reduction='mean'):
        super(FocalLoss3dMap, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.generate_map = generate_gaussian_heatmap_batch

    def forward(self, predictions, targets):
        targets = self.generate_map(targets)
        # print(targets[targets==1].sum())
        losses = torch.where(targets == 1,
                             - (1 - predictions) ** self.alpha * torch.log(predictions + 1e-6),
                             - (1 - targets) ** self.beta * predictions ** self.alpha * torch.log(1 - predictions + 1e-6))
        # print(losses.shape, torch.sum(losses, dim=[1, 2, 3, 4]))
        if self.reduction == 'mean':
            batch_sum = torch.sum(losses, dim=[1, 2, 3])
            return torch.mean(batch_sum)
        elif self.reduction == 'sum':
            return torch.sum(losses)
        else:
            return losses


def generate_gaussian_heatmap_batch(centers_sizes, sigma=6, grid_size=(128, 128, 128)):
    """
    Generate a batch of 3D Gaussian heatmaps from a batch of centers and sizes.
    :param centers_sizes: Tensor of shape (batch_size, 6) containing [x, y, z, w, h, d] for each sample
    :param sigma: Gaussian distribution's sigma
    :param grid_size: tuple(width, height, depth) of the heatmap grid
    :return: Batch of 3D heatmaps as a tensor
    """
    # Extract centers and sizes
    centers = centers_sizes[:, :3]  # Get x, y, z for each batch
    # sizes = centers_sizes[:, 3:]  # Not directly used in Gaussian, but could be used to adjust sigma

    # Create meshgrid
    z = torch.arange(0, grid_size[2], dtype=torch.float32, device=centers.device)
    y = torch.arange(0, grid_size[1], dtype=torch.float32, device=centers.device)
    x = torch.arange(0, grid_size[0], dtype=torch.float32, device=centers.device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    # print(zz.shape, yy.shape, xx.shape)

    # Expand grid to batch size
    xx = xx.expand(centers.size(0), *xx.size())
    yy = yy.expand(centers.size(0), *yy.size())
    zz = zz.expand(centers.size(0), *zz.size())

    # Subtract center coordinates for the Gaussian formula
    cx, cy, cz = centers[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), \
        centers[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), \
        centers[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # Calculate the heatmap using Gaussian formula
    heatmap = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / (2 * sigma ** 2))

    return heatmap
if __name__ == '__main__':
    # loss = FocalLoss(alpha=[1-0.67, 1-0.28, 1-0.025, 1-0.015])
    inputs = torch.sigmoid(torch.randn(1, 1, 128, 128, 128))
    # targets = torch.randint(1, 6)
    targets = torch.tensor([[3, 4, 3, 2, 2, 2]])
    # print(targets)
    loss = FocalLoss3dMap()
    print(loss(inputs, targets))
