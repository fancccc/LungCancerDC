from monai.networks.nets import ViT
from torch import nn

class COMP_VIT(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.net = ViT(in_channels=1, img_size=img_size, patch_size=(8, 8, 8), proj_type='conv', pos_embed_type='sincos', classification=True, num_classes=num_classes)
    def forward(self, img):
        x, _ = self.net(img)
        return x
if __name__ == '__main__':
    import torch
    net = COMP_VIT(img_size=(32, 32, 32), num_classes=3)

    x = net(torch.rand(1, 1, 32, 32, 32))
    print(x)
