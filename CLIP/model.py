from CLIP.CLIP_VBCR import CLIP3 as CLIP
from CLIP.CLIP_VBCR import CLIP4
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba_ssm import Mamba
from CLIP.classifier import DenseNet
from src.loss import CosineSimilarityLoss, FocalLoss, ClipLoss
class CLIP_VBCRNet(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(CLIP_VBCRNet, self).__init__()
        self.CLIP1 = CLIP(embed_dim=32, hidden_size=768, img_size=(128, 128, 32), vision_patch_size=8, context_length=6,
                          transformer_heads=12, transformer_layers=6)
        self.CLIP2 = CLIP(embed_dim=32, hidden_size=768, img_size=(32, 32, 32), vision_patch_size=8, context_length=6,
                          transformer_heads=12, transformer_layers=6)

        self.Mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=hidden_size, # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,    # Local convolution width
                            expand=2,    # Block expansion factor
                            )

        self.classifier = DenseNet(in_channels=8, classes=num_classes)

        self.loss1 = CosineSimilarityLoss()
        self.weight = 0.5
        self.focal_loss_weight = [1-0.5347, 1-0.2410, 1-0.1852, 1-0.039]
        self.loss2 = FocalLoss(alpha=self.focal_loss_weight)


    def forward(self, data):
        ct128, ct32, bbox128, bbox32 = data['ct128'], data['ct32'], data['bbox128'], data['bbox32']
        data['image'], data['bbox'] = ct128, bbox128
        cosine_similarity1, feats1 = self.CLIP1(data)
        # print(feats1, feats1[0].shape)
        data['image'], data['bbox'] = ct32, bbox32
        cosine_similarity2, feats2 = self.CLIP2(data)
        feats = feats1 + feats2
        feats = [i.unsqueeze(1) for i in feats]
        feats = torch.cat(feats, dim=1)
        # print(feats)
        feats = self.Mamba(feats)
        out = self.classifier(feats)
        # print(torch.isnan(out).sum())
        loss11 = self.weight*self.loss1(cosine_similarity1, data['label'])
        loss12 = self.weight*self.loss1(cosine_similarity2, data['label'])
        loss1 = loss11 + loss12
        # print(loss11)
        loss2 = self.loss2(out, data['label'])
        # print(f'loss:{loss1}, {loss2}')
        loss = 0.8*loss1 + 0.2*loss2
        return out, loss

class CLIP_VBCRNet2(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(CLIP_VBCRNet2, self).__init__()
        self.CLIP1 = CLIP(embed_dim=32, hidden_size=768, img_size=(128, 128, 32), vision_patch_size=8, context_length=6,
                          transformer_heads=12, transformer_layers=6)
        self.CLIP2 = CLIP(embed_dim=32, hidden_size=768, img_size=(32, 32, 32), vision_patch_size=8, context_length=6,
                          transformer_heads=12, transformer_layers=6)

        self.Mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=hidden_size, # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,    # Local convolution width
                            expand=2,    # Block expansion factor
                            )

        self.classifier = DenseNet(in_channels=8, classes=num_classes)

        self.loss1 = ClipLoss()
        self.weight = 0.5
        self.focal_loss_weight = [1-0.5347, 1-0.2410, 1-0.1852, 1-0.039]
        self.loss2 = FocalLoss(alpha=self.focal_loss_weight)


    def forward(self, data):
        ct128, ct32, bbox128, bbox32 = data['ct128'], data['ct32'], data['bbox128'], data['bbox32']
        data['image'], data['bbox'] = ct128, bbox128
        cosine_similarity1, feats1 = self.CLIP1(data)
        # print(feats1, feats1[0].shape)
        data['image'], data['bbox'] = ct32, bbox32
        cosine_similarity2, feats2 = self.CLIP2(data)
        feats = feats1 + feats2
        feats = [i.unsqueeze(1) for i in feats]
        feats = torch.cat(feats, dim=1)
        # print(feats)
        feats = self.Mamba(feats)
        out = self.classifier(feats)
        # print(torch.isnan(out).sum())
        loss11 = self.weight*self.loss1(cosine_similarity1)
        loss12 = self.weight*self.loss1(cosine_similarity2)
        loss1 = loss11 + loss12
        # print(loss11)
        loss2 = self.loss2(out, data['label'])
        # print(f'loss:{loss1}, {loss2}')
        loss = (loss1 + loss2) / 2
        return out, loss
class CLIP_VBCRNet3(CLIP_VBCRNet2):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.classifier = DenseNet(in_channels=1368, classes=num_classes, layer_num=[2, 2, 2, 2])
        self.loss2 = nn.CrossEntropyLoss()

    def forward(self, data):
        ct128, ct32, bbox128, bbox32 = data['ct128'], data['ct32'], data['bbox128'], data['bbox32']
        data['image'], data['bbox'] = ct128, bbox128
        cosine_similarity1, feats1 = self.CLIP1(data)
        # print(feats1, feats1[0].shape)
        data['image'], data['bbox'] = ct32, bbox32
        cosine_similarity2, feats2 = self.CLIP2(data)
        feats = torch.cat((feats1, feats2), dim=1)
        # print('feats:', feats.shape)
        # feats = [i.unsqueeze(1) for i in feats]
        # feats = torch.cat(feats, dim=1)
        # print(feats)
        feats = self.Mamba(feats)
        out = self.classifier(feats)
        # print(torch.isnan(out).sum())
        loss11 = self.weight*self.loss1(cosine_similarity1)
        loss12 = self.weight*self.loss1(cosine_similarity2)
        loss1 = loss11 + loss12
        # print(loss11)
        loss2 = self.loss2(out, data['label'])
        # print(f'loss:{loss1}, {loss2}')
        loss = 0.3*loss1 + 0.7*loss2
        return out, loss


class CLIP_VBCRNet4(CLIP_VBCRNet2):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.classifier = DenseNet(in_channels=1305, classes=num_classes, layer_num=[2, 2, 2, 2])
        self.loss2 = nn.CrossEntropyLoss()
        self.CLIP2 = CLIP4(embed_dim=32, hidden_size=768, img_size=(32, 32, 32), vision_patch_size=8, context_length=6,
            transformer_heads=12, transformer_layers=6)

    def forward(self, data):
        ct128, ct32, bbox128, bbox32 = data['ct128'], data['ct32'], data['bbox128'], data['bbox32']
        data['image'], data['bbox'] = ct128, bbox128
        cosine_similarity1, feats1 = self.CLIP1(data)
        # print(feats1, feats1[0].shape)
        data['image'], data['bbox'] = ct32, bbox32
        cosine_similarity2, feats2 = self.CLIP2(data)
        feats = torch.cat((feats1, feats2), dim=1)
        # print('feats:', feats.shape)
        # feats = [i.unsqueeze(1) for i in feats]
        # feats = torch.cat(feats, dim=1)
        # print(feats)
        feats = self.Mamba(feats)
        # print(feats.shape)
        out = self.classifier(feats)
        # print(torch.isnan(out).sum())
        loss11 = self.weight*self.loss1(cosine_similarity1)
        loss12 = self.weight*self.loss1(cosine_similarity2)
        loss1 = loss11 + loss12
        # print(loss11)
        loss2 = self.loss2(out, data['label'])
        # print(f'loss:{loss1}, {loss2}')
        loss = 0.3*loss1 + 0.7*loss2
        return out, loss

from CLIP.CLIP_VBCR import CLIP_C3_V1
from CLIP.multiModolAtt import MultiModalAtt
from src.nets import ChannelAttention
class CLIP_VBCRNet5(nn.Module):
    '''
    USE DATA: CT32, CT128, EHR, BBOX128
    '''
    def __init__(self, hidden_size=768, num_classes=2):
        super(CLIP_VBCRNet5, self).__init__()
        self.CLIP1 = CLIP_C3_V1(embed_dim=32, hidden_size=768, img_encode_type='vit', img_size=(128, 128, 32), vision_patch_size=8,
                          context_length=6,
                          transformer_heads=12, transformer_layers=6)
        self.CLIP2 = CLIP_C3_V1(embed_dim=32, hidden_size=768, img_encode_type='resnet18', img_size=(32, 32, 32), vision_patch_size=8, context_length=6,
            transformer_heads=12, transformer_layers=6)

        # self.Mamba = Mamba(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     d_model=hidden_size,  # Model dimension d_model
        #     d_state=16,  # SSM state expansion factor
        #     d_conv=4,  # Local convolution width
        #     expand=2,  # Block expansion factor
        # )
        self.att = MultiModalAtt()
        self.ca = ChannelAttention(in_channels=1091)
        # self.classifier = DenseNet(in_channels=1091, classes=num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.loss1 = ClipLoss()
        self.weight = 0.5
        self.focal_loss_weight = [1 - 0.5347, 1 - 0.2410, 1 - 0.1852 - 0.039]
        self.loss2 = FocalLoss(alpha=self.focal_loss_weight)

    def forward(self, data):
        ct128, ct32, bbox128, bbox32 = data['ct128'], data['ct32'], data['bbox128'], data['bbox32']
        data['image'], data['bbox'] = ct128, bbox128
        cosine_similarity1, feats1 = self.CLIP1(data)
        # print(feats1, feats1[0].shape)
        data['image'], data['bbox'] = ct32, bbox32
        cosine_similarity2, feats2 = self.CLIP2(data)
        feats = torch.cat((feats1, feats2), dim=1)
        feats, _ = self.att(feats)
        # print('feats:', feats.shape)
        # feats = [i.unsqueeze(1) for i in feats]
        # feats = torch.cat(feats, dim=1)
        # print(feats)
        # feats = self.Mamba(feats)
        # print(feats.shape)
        out = self.ca(feats)
        out = out.mean(dim=1)
        # print(out.shape)
        out = self.classifier(out)
        # print(torch.isnan(out).sum())
        loss11 = self.weight*self.loss1(cosine_similarity1)
        loss12 = self.weight*self.loss1(cosine_similarity2)
        loss1 = loss11 + loss12
        # print(loss11)
        loss2 = self.loss2(out, data['label'])
        # print(f'loss:{loss1}, {loss2}')
        loss = 0.3*loss1 + 0.7*loss2
        return out, loss


if __name__ == '__main__':
    from src.dataloader import split_pandas, LungDataset, DataLoader
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
    model = CLIP_VBCRNet5(num_classes=3).to("cuda")
    # ct128 = torch.randn(1, 1, 128, 128, 32)
    # ct32 = torch.randn(1, 1, 32, 32, 32)
    # bbox128 = torch.tensor([[64, 64, 64, 23, 22, 21]])
    # bbox32 = torch.tensor([[25, 25, 25, 23, 22, 21]])
    # radiomics = torch.randn(1, 107, 1)
    # clinicals = torch.randn(1, 27, 1)
    # label = torch.tensor([1], dtype=torch.long)
    # data = {'ct128': ct128, 'ct32': ct32, 'bbox128': bbox128, 'bbox32': bbox32, 'clinical': clinicals, 'radiomics': radiomics, 'label': label}
    for data in train_loader:
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to('cuda')
        # print(data['bbox32'], data['bbox128'], data['label'])
        out, loss = model(data)
        print(out, loss)
        break

