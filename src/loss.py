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

## SAMCLIP
class ClassificationLoss(nn.Module):
    def __init__(self, class_weights):
        super(ClassificationLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, predicted_probabilities, ground_truth_labels):
        loss = -torch.sum(
            self.class_weights[0] * (ground_truth_labels == 1) * torch.log(predicted_probabilities[:, 1] + 1e-8))
        loss -= torch.sum(
            self.class_weights[1] * (ground_truth_labels == 0) * torch.log(predicted_probabilities[:, 0] + 1e-8))
        return loss

class MultiTaskAwareLoss(nn.Module):
    def __init__(self):
        super(MultiTaskAwareLoss, self).__init__()

    def forward(self, predicted_probabilities, segmentation_output):
        softmax = nn.Softmax(dim=1)
        predicted_probabilities = softmax(predicted_probabilities)
        p = predicted_probabilities[:, 1]  # Probability for positive class
        s = segmentation_output

        js_divergence = torch.sum(p * torch.log(p / (s + 1e-8) + 1e-8))
        return js_divergence

class TotalLoss(nn.Module):
    def __init__(self, alpha, beta, class_weights):
        super(TotalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # self.segmentation_loss = SegmentationLoss()
        self.classification_loss = ClassificationLoss(class_weights)
        self.multi_task_loss = MultiTaskAwareLoss()

    def forward(self, predicted_masks, ground_truth_masks, predicted_probabilities, ground_truth_labels):
        # seg_loss = self.segmentation_loss(predicted_masks, ground_truth_masks)
        cls_loss = self.classification_loss(predicted_probabilities, ground_truth_labels)
        mt_loss = self.multi_task_loss(predicted_probabilities, predicted_masks)

        # total_loss = mt_loss + self.alpha * seg_loss + self.beta * cls_loss
        total_loss = mt_loss + self.beta * cls_loss
        return total_loss

class CosineSimilarityLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(CosineSimilarityLoss, self).__init__()
        self.alpha = alpha  # 正样本相似度的权重
        self.beta = beta    # 负样本相似度的权重

    def forward(self, similarity_matrices, labels):
        losses = []
        for matrix in similarity_matrices:
            batch_size = matrix.size(0)

            # 创建正样本掩码（包含自身）
            labels_expanded = labels.unsqueeze(1)
            positive_mask = (labels_expanded == labels_expanded.t()).float()
            # 负样本掩码
            negative_mask = (labels_expanded != labels_expanded.t()).float()
            # 损失计算
            positive_loss = -self.alpha * (matrix * positive_mask).sum() / (positive_mask.sum() + 1e-8)
            negative_loss = self.beta * (matrix * negative_mask).sum() / (negative_mask.sum() + 1e-8)

            if torch.isnan(negative_loss):
                loss = positive_loss
            else:
            # 总损失
                loss = positive_loss + negative_loss
            loss = torch.relu(loss)
            losses.append(loss)

        # 所有相似度矩阵的平均损失
        total_loss = torch.stack(losses).mean()
        return total_loss

class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # 生成一个 [0, 1, 2, ..., num_logits-1] 的标签张量
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        # labels = torch.randint(0, 2, (1, num_logits), device=device, dtype=torch.long)
        # print(labels)
        return labels

    def forward(self, logits, output_dict=False):
        device = logits[0][0].device
        total_loss = 0
        # print('logits len: ', len(logits))
        for logit in logits:
            logits_per_image, logits_per_text = logit

            labels = self.get_ground_truth(device, logits_per_image.shape[0])

            total_loss += (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2
        total_loss /= len(logits)
        return {"contrastive_loss": total_loss} if output_dict else total_loss
if __name__ == '__main__':
    # # loss = FocalLoss(alpha=[1-0.67, 1-0.28, 1-0.025, 1-0.015])
    # inputs = torch.sigmoid(torch.randn(1, 1, 128, 128, 128))
    # # targets = torch.randint(1, 6)
    # targets = torch.tensor([[3, 4, 3, 2, 2, 2]])
    # # print(targets)
    # loss = FocalLoss3dMap()
    # print(loss(inputs, targets))
    clip = ClipLoss()
    similarity_matrices = [[torch.randn(10, 10) for _ in range(2)]]  # 示例余弦相似度矩阵
    print(similarity_matrices)
    labels = torch.randint(0, 3, (10, 1))  # 示例标签

    # 初始化损失函数
    # cosine_loss = CosineSimilarityLoss()

    # 计算损失
    loss = clip(similarity_matrices)
    print("Cosine Similarity Loss:", loss.item())