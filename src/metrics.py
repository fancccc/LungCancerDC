import torch
def bbox_3d_iou(batch_box1, batch_box2):
    # 将中心坐标和尺寸转换为最小和最大坐标
    box1_min = batch_box1[:, :3] - 0.5 * batch_box1[:, 3:]
    box1_max = batch_box1[:, :3] + 0.5 * batch_box1[:, 3:]
    box2_min = batch_box2[:, :3] - 0.5 * batch_box2[:, 3:]
    box2_max = batch_box2[:, :3] + 0.5 * batch_box2[:, 3:]

    # 计算两个边界框批次的交集
    min_point = torch.max(box1_min, box2_min)
    max_point = torch.min(box1_max, box2_max)
    intersection = torch.prod(torch.clamp(max_point - min_point, min=0), dim=1)

    # 计算两个边界框批次的并集
    box1_volume = torch.prod(batch_box1[:, 3:], dim=1)
    box2_volume = torch.prod(batch_box2[:, 3:], dim=1)
    union = box1_volume + box2_volume - intersection

    # 计算IoU
    iou = intersection / (union + 1e-6)  # 添加小常数避免除以零
    return iou.mean()

def center_distance(preds, targets, size=1):
    eu_dist = torch.sqrt(torch.sum((preds[:, :3] - targets[:, :3]) ** 2, dim=1))
    norm_dist = eu_dist / size
    return norm_dist.mean()

def find_max_points(heatmaps):
    """
    Find the coordinates of the maximum points in a batch of 3D heatmaps.
    :param heatmaps: 4D tensor of shape (batch_size, depth, height, width)
    :return: List of coordinates of the maximum points for each heatmap in the batch
    """
    max_points_list = []
    for heatmap in heatmaps:
        # Flatten the heatmap to find the index of the maximum value
        max_val_index = heatmap.argmax()

        # Convert the flat index back to 3D coordinates
        depth = max_val_index // (heatmap.shape[1] * heatmap.shape[2])
        height = (max_val_index % (heatmap.shape[1] * heatmap.shape[2])) // heatmap.shape[2]
        width = (max_val_index % (heatmap.shape[1] * heatmap.shape[2])) % heatmap.shape[2]

        max_points_list.append([depth.item(), height.item(), width.item(), 20, 20, 20])
    max_points_list = torch.tensor(max_points_list, device=heatmaps.device)
    return max_points_list

if __name__ == '__main__':
    #    batch_size = 2
    #    batch_box1 = torch.tensor([[57.606216, 64.076935, 78.13548,  20.973412, 20.380959, 21.509022],
    # [56.076504, 62.371883, 76.05308,  20.418339, 19.842987, 20.937244]])
    #    batch_box2 = torch.tensor([[ 32.5, 118.5, 113.,   28.,   36.,   36. ],
    # [ 85.,   54.,   67.,   21.,   21.,   21. ]])
    #    batch_box1 = torch.tensor([[31.5, 118.5, 113., 28., 36., 36.],
    #                               [85., 54., 67., 21., 21., 21.]])
    #    iou = bbox_3d_iou(batch_box1, batch_box2)
    #    print(f"Batch 3D IoU: {iou.mean()}")
    #
    #    pred_centers = torch.tensor([[60, 2.0, 3.0, 2],
    #                                 [4.0, 5.0, 6.0, 5]])
    #    target_centers = torch.tensor([[1.1, 2.2, 3.3, 8],
    #                                   [3.9, 5.0, 6.1, 7]])
    #    norm_distances = center_distance(pred_centers, target_centers)
    #    print(norm_distances)
    inputs = torch.sigmoid(torch.randn(1, 1, 128, 128, 128))
    # targets = torch.randint(1, 6)
    targets = torch.tensor([[30, 20, 30, 2, 2, 2]])
    inputs = find_max_points(inputs)
    print(inputs, targets)
    dist = center_distance(inputs, targets)
    print(dist)

