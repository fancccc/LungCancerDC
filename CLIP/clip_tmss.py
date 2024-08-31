# -*- coding: utf-8 -*-
'''
@file: clip_tmss.py
@author: fanc
@time: 2024/8/27 23:01
'''

import torch
from typing import Sequence, Union, Tuple
import torch
import torch.nn as nn
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from CLIP.CLIP_VBCR import TabularTransformer
from CLIP.multiModolAtt import MultiModalAtt
from src.nets import ChannelAttention
from src.loss import CosineSimilarityLoss, FocalLoss, ClipLoss
n_clin_var = 6
# __all__ = ["ViT"]

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_layers: int = 12,
            num_heads: int = 12,
            pos_embed: str = "conv",
            classification: bool = False,
            num_classes: int = 2,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())

        ## Projection of EHR
        self.EHR_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):

        x = self.patch_embedding(x)  # img, clin_var = x

        if self.classification:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if self.classification:
            x = self.classification_head(x[:, 0])

        return x, hidden_states_out

    # Copyright 2020 - 2021 MONAI Consortium


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.layers import Conv
from monai.utils import ensure_tuple_rep, optional_import
from monai.utils.module import look_up_option

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}
class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Example::

        >>> from monai.networks.blocks import PatchEmbeddingBlock
        >>> PatchEmbeddingBlock(in_channels=4, img_size=32, patch_size=8, hidden_size=32, num_heads=4, pos_embed="conv")

    """

    def __init__(
            self,
            in_channels: int,
            img_size: Union[Sequence[int], int],
            patch_size: Union[Sequence[int], int],
            hidden_size: int,
            num_heads: int,
            pos_embed: str,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.pos_embed = look_up_option(pos_embed, SUPPORTED_EMBEDDING_TYPES)

        # img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)]) + 1  # +1 for EHR
        # print('n patch', self.n_patches, img_size)
        self.patch_dim = in_channels * np.prod(patch_size)

        self.patch_embeddings: nn.Module
        if self.pos_embed == "conv":
            self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size
            )
        elif self.pos_embed == "perceptron":
            # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"
            chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
            from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
            to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
            axes_len = {f"p{i + 1}": p for i, p in enumerate(patch_size)}
            self.patch_embeddings = nn.Sequential(
                Rearrange(f"{from_chars} -> {to_chars}", **axes_len), nn.Linear(self.patch_dim, hidden_size)
            )

        self.EHR_proj = nn.Sequential(nn.Linear(n_clin_var, hidden_size))

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trunc_normal_(self, tensor, mean, std, a, b):
        # From PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    def forward(self, x):
        img, clin_var = x
        x = self.patch_embeddings(img)
        # print(clin_var.shape)
        clin_var = self.EHR_proj(clin_var)
        clin_var = clin_var.unsqueeze(dim=1)

        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)

        x = torch.cat([clin_var, x], dim=1)
        # print(x.shape, self.position_embeddings.shape)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class TmssModule(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            img_size: int = 32,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "conv",
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            patch_size: int = 8,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size

        self.classification = False
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, data):
        img, cli = data['image'], data['bbox']
        x = (img, cli)
        x, _ = self.vit(x)
        x = torch.mean(x, dim=1, keepdim=True)
        # # print(x.shape)
        # classification_output = self.fc(x)
        return x
class CLIP_C3_V2(nn.Module):
    """
    clip with tmss
    ct+bbox
    ehr
    """
    def __init__(self,
                 embed_dim: int,
                 hidden_size: 768,
                 # vision
                 img_size: Tuple[int, int, int],
                 vision_patch_size: 8,
                 # text
                 # context_length: int,
                 transformer_heads: 12,
                 transformer_layers: 6
                 ):
        super().__init__()

        self.encode_image = TmssModule(img_size=img_size)
        # self.context_length = context_length

        self.encode_clinical = TabularTransformer(
                                            embed_dim=embed_dim,
                                            context_length=27,
                                            vocab_size=27,
                                            transformer_width=hidden_size,
                                            transformer_heads=transformer_heads,
                                            transformer_layers=transformer_layers)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, data):
        image, bbox, clinical = data['image'], data['bbox'], data['clinical']
        image_features = self.encode_image(data)
        # image_features = self.encode_image(data)
        # image_features = image_features.squeeze(dim=-1).squeeze(dim=-1).permute(0, 2, 1)
        # print(image_features.shape)
        # image_features = self.proj_image(image_features)
        # print(image_features.shape)
        # image_features = self.proj_image(image_features)
        # print(image_features.shape)
        # print(bbox_features.shape)
        clinical_features, _ = self.encode_clinical(clinical.unsqueeze(dim=-1))
        # print(clinical_features.shape)
        feats = torch.cat([image_features, clinical_features], dim=1)
        # print(radiomics_features.shape, clinical_features.shape)
        # print('#'*20)
        # print(image_features, '\n', bbox_features, '\n',radiomics_features, '\n', clinical_features)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # bbox_features = bbox_features / bbox_features.norm(dim=1, keepdim=True)
        # radiomics_features = radiomics_features / radiomics_features.norm(dim=1, keepdim=True)
        clinical_features = clinical_features / clinical_features.norm(dim=1, keepdim=True)
        image_features = image_features.mean(dim=1, keepdim=False)
        # bbox_features = bbox_features.mean(dim=1, keepdim=False)
        # radiomics_features = radiomics_features.mean(dim=1, keepdim=False)
        clinical_features = clinical_features.mean(dim=1, keepdim=False)
        # cosine similarity as logits
        # Calculate cosine similarity across different modalities
        logit_scale = self.logit_scale.exp()
        # logits_image_bbox = logit_scale * image_features @ bbox_features.t()
        # logits_image_radiomics = logit_scale * image_features @ radiomics_features.t()
        logits_image_clinical = logit_scale * image_features @ clinical_features.t()

        # shape = [global_batch_size, global_batch_size]
        logits = [#[logits_image_bbox, logits_image_bbox.t()],
                  # [logits_image_radiomics, logits_image_radiomics.t()],
                  [logits_image_clinical, logits_image_clinical.t()]]
        # feats = [image_features, bbox_features, radiomics_features, clinical_features]
        return logits, feats
class TMSSNet(nn.Module):
    def __init__(
            self,
            num_classes: int,
            in_channels: int = 1,
            # out_channels: int,
            img_size: int = 32,
            # feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "conv",
            # norm_name: Union[Tuple, str] = "instance",
            # conv_block: bool = True,
            # res_block: bool = True,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            patch_size: int = 8,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size

        self.classification = False
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.fc = nn.Sequential(nn.Linear(hidden_size, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(256, 3))

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, data):
        img, cli = data['ct32'], data['clinical']
        x = (img, cli)
        # print(x[0].shape, x[1].shape)
        # img_in, clin_in = x
        # Image path
        # img = self.img_patch_embedding(img_in)  # 216*h
        # img_clin = self.patch_embedding(x)  # 217*h
        # clinical and img out the end layer : x
        x, _ = self.vit(x)
        x = torch.mean(x, dim=1)
        # print(x.shape)
        classification_output = self.fc(x)

        return classification_output

class CLIP_TMSS_Net(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.CLIP1 = CLIP_C3_V2(embed_dim=128, hidden_size=768, img_size=(32, 32, 32), vision_patch_size=8,
                          transformer_heads=12, transformer_layers=6)
        self.CLIP2 = CLIP_C3_V2(embed_dim=128, hidden_size=768, img_size=(128, 128, 32), vision_patch_size=8,
                          transformer_heads=12, transformer_layers=6)
        self.att = MultiModalAtt()
        self.ca = ChannelAttention(in_channels=56)

        # self.classifier = nn.Sequential(
        #     nn.Linear(768, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, num_classes)
        # )
        self.loss1 = ClipLoss()
        self.weight = 0.5
        self.focal_loss_weight = [1 - 0.5347, 1 - 0.2410, 1 - 0.1852 - 0.039]
        self.loss2 = FocalLoss(alpha=self.focal_loss_weight)
        self.classifier = nn.Sequential(
                                nn.Linear(768, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(256, num_classes))

    def forward(self, data):
        ct128, ct32, bbox128, bbox32 = data['ct128'], data['ct32'], data['bbox128'], data['bbox32']
        data['image'], data['bbox'] = ct32, bbox32
        cosine_similarity1, feats1 = self.CLIP1(data)
        # print(feats1, feats1[0].shape)
        data['image'], data['bbox'] = ct128, bbox128
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
        loss11 = self.weight * self.loss1(cosine_similarity1)
        loss12 = self.weight * self.loss1(cosine_similarity2)
        loss1 = loss11 + loss12
        # print(loss11)
        #
        loss2 = self.loss2(out, data['label'])
        # print(f'loss:{loss1}, {loss2}')
        loss = 0.3 * loss1 + 0.7 * loss2
        return out, loss

if __name__ == "__main__":
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
    # model = CLIP_C3_V2(embed_dim=128,
    #                    img_size=(128, 128, 32),
    #                    hidden_size=768,
    #                    vision_patch_size=8,
    #                    transformer_heads=12,
    #                    transformer_layers=6
    #                    )
    # model = TMSSNet(in_channels=1, img_size=32)
    # model = PatchEmbeddingBlock(
    #         in_channels=1,
    #         img_size=(128, 128, 32),
    #         patch_size=8,
    #         hidden_size=768,
    #         num_heads=12,
    #         pos_embed="conv",
    #         dropout_rate=0,
    #         spatial_dims=3,
    #     )
    model = CLIP_TMSS_Net(num_classes=3).to("cuda")
    for data in train_loader:
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to('cuda')
        out, loss = model(data)
        print(out, loss)
        break

    # img = torch.randn(1, 1, 128, 128, 32)
    # bbox = torch.tensor([[0.4, 0.4, 0.4, 0.1, 0.1, 0.1]])
    # cli = torch.randn(1, 27)
    # data = {'image': img, 'bbox': bbox, 'clinical': cli}
    # out = model(data)


