import SimpleITK as sitk
import numpy as np
import os

path = "/home/zcd/datasets/LungCancer/cropped_masks_256"
files = os.listdir(path)
c = 0
for file in files:
    label_image = sitk.ReadImage(os.path.join(path, file))
    # 将SimpleITK图像转换为NumPy数组，检查标签1是否存在
    label_array = sitk.GetArrayFromImage(label_image)
    if np.any(label_array == 1):
        continue
    else:
        c += 1
        print(file, c)


    # # 如果标签存在，正确执行LabelShapeStatisticsImageFilter
    # label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    # label_shape_filter.Execute(label_image)
    #
    # # 如果标签1存在，尝试获取边界框
    # if label_shape_filter.HasLabel(1):
    #     bounding_box = label_shape_filter.GetBoundingBox(1)
    #     # print("边界框:", bounding_box)
    # else:
    #     print("没有找到标签1的边界框。")