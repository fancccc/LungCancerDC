from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# from monai.networks.nets.vit import ViT
from CLIP.VIT import ViT
# from src.nets2 import RadiomicsTransformer, MLP
from src.nets import generate_model
#This part is derived and modified by : https://github.com/openai/CLIP/blob/main/clip/model.py
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class TextTransformer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()
        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # if isinstance(self.visual, ModifiedResNet):
        #     if self.visual.attnpool is not None:
        #         std = self.visual.attnpool.c_proj.in_features ** -0.5
        #         nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
        #
        #     for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
        #         for name, param in resnet_block.named_parameters():
        #             if name.endswith("bn3.weight"):
        #                 nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x, x @ self.text_projection
class TabularTransformer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()
        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Linear(1, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # if isinstance(self.visual, ModifiedResNet):
        #     if self.visual.attnpool is not None:
        #         std = self.visual.attnpool.c_proj.in_features ** -0.5
        #         nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
        #
        #     for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
        #         for name, param in resnet_block.named_parameters():
        #             if name.endswith("bn3.weight"):
        #                 nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # print('token embedding:',x.shape)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x, x @ self.text_projection

class TabularEncoder(nn.Module):
    def __init__(self,
                 context_length: int,
                 hid_width: int=768):
        super().__init__()
        # self.context_length = context_length
        self.fc = nn.Sequential(
            nn.Linear(1, hid_width),
            # nn.BatchNorm1d(hid_width),
            # nn.ReLU(),
            # nn.Dropout(p=0.1)
        )
        # self.fc2 = nn.Linear(context_length, 1)
        # self.out_layer = nn.Linear(context_length, hid_width)
        # self.token_embedding = nn.Linear(1, hid_width)
        # self.positional_embedding = nn.Parameter(torch.empty(self.context_length, hid_width))
        # self.positional_embedding = nn.Parameter(torch.empty(context_length))
        # self.initialize_parameters()
    def initialize_parameters(self):
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=1)

    def forward(self, text):
        x = text
        # print(x.shape)
        # x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # print('token embedding:',x.shape)
        # print(text.shape, self.positional_embedding.shape)
        # x = text + self.positional_embedding
        # print(x)
        # print(x.shape, self.positional_embedding.shape)
        x = self.fc(x)
        x = torch.mean(x, dim=1, keepdim=True)
        # x = self.fc2(x.permute(0, 2, 1))
        # x = x.permute(0, 2, 1)
        # print(x)
        # x = self.out_layer(x.squeeze(-1))
        # x = x.unsqueeze(1)
        # print(x.shape)
        return x
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 hidden_size: 768,
                 # vision
                 img_size: Tuple[int, int, int],
                 vision_patch_size: 8,
                 # text
                 context_length: int,
                 transformer_heads: 12,
                 transformer_layers: 6
                 ):
        super().__init__()
        self.context_length = context_length
        self.encode_image = ViT(
                            in_channels=1,
                            img_size=img_size,
                            patch_size=vision_patch_size,
                            hidden_size=hidden_size,
                            mlp_dim=3072,
                            num_layers=transformer_layers,
                            num_heads=transformer_heads)
        self.encode_bbox = TextTransformer(
                            embed_dim=embed_dim,
                            context_length=context_length,
                            vocab_size=1024,
                            transformer_width=hidden_size,
                            transformer_heads=transformer_heads,
                            transformer_layers=transformer_layers)
        # self.encode_radiomics = MLP(input_dim=107, hidden_dim=1024, num_layers=3, output_dim=hidden_size)
        # self.encode_clinical = MLP(input_dim=27, hidden_dim=1024, num_layers=3, output_dim=hidden_size)
        # self.encode_clinical = RadiomicsTransformer(
        #                     num_features=hidden_size,
        #                     num_layers=transformer_layers,
        #                     num_heads=transformer_heads)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, data):
        image, bbox, radiomics, clinical = data['image'], data['bbox'], data['radiomics'], data['clinical']
        image_features, _ = self.encode_image(image)
        bbox_features, _ = self.encode_bbox(bbox)
        radiomics_features = self.encode_radiomics(radiomics)
        clinical_features = self.encode_clinical(clinical)
        # print(radiomics_features.shape, clinical_features.shape)
        # print('#'*20)
        # print(image_features, '\n', bbox_features, '\n',radiomics_features, '\n', clinical_features)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        bbox_features = bbox_features / bbox_features.norm(dim=1, keepdim=True)
        radiomics_features = radiomics_features / radiomics_features.norm(dim=1, keepdim=True)
        clinical_features = clinical_features / clinical_features.norm(dim=1, keepdim=True)
        image_features = image_features.mean(dim=1, keepdim=False)
        bbox_features = bbox_features.mean(dim=1, keepdim=False)
        # radiomics_features = radiomics_features.mean(dim=1, keepdim=False)
        # clinical_features = clinical_features.mean(dim=1, keepdim=False)


        # cosine similarity as logits
        # Calculate cosine similarity across different modalities
        logit_scale = self.logit_scale.exp()
        logits_image_bbox = logit_scale * image_features @ bbox_features.t()
        logits_image_radiomics = logit_scale * image_features @ radiomics_features.t()
        logits_image_clinical = logit_scale * image_features @ clinical_features.t()

        # shape = [global_batch_size, global_batch_size]
        cosine_similarity = [logits_image_bbox, logits_image_radiomics, logits_image_clinical]
        feats = [image_features, bbox_features, radiomics_features, clinical_features]
        return cosine_similarity, feats


class CLIP2(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 hidden_size: 768,
                 # vision
                 img_size: Tuple[int, int, int],
                 vision_patch_size: 8,
                 # text
                 context_length: int,
                 transformer_heads: 12,
                 transformer_layers: 6
                 ):
        super().__init__()
        self.context_length = context_length
        self.encode_image = ViT(
                            in_channels=1,
                            img_size=img_size,
                            patch_size=vision_patch_size,
                            hidden_size=hidden_size,
                            mlp_dim=3072,
                            num_layers=transformer_layers,
                            num_heads=transformer_heads)
        self.encode_bbox = TextTransformer(
                            embed_dim=embed_dim,
                            context_length=context_length,
                            vocab_size=1024,
                            transformer_width=hidden_size,
                            transformer_heads=transformer_heads,
                            transformer_layers=transformer_layers)
        # self.encode_radiomics = MLP(input_dim=107, hidden_dim=1024, num_layers=3, output_dim=hidden_size)
        # self.encode_clinical = MLP(input_dim=27, hidden_dim=1024, num_layers=3, output_dim=hidden_size)
        self.encode_radiomics = TabularTransformer(
                                            embed_dim=embed_dim,
                                            context_length=107,
                                            vocab_size=107,
                                            transformer_width=768,
                                            transformer_heads=12,
                                            transformer_layers=6)
        self.encode_clinical = TabularTransformer(
                                            embed_dim=32,
                                            context_length=27,
                                            vocab_size=27,
                                            transformer_width=768,
                                            transformer_heads=12,
                                            transformer_layers=6)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, data):
        image, bbox, radiomics, clinical = data['image'], data['bbox'], data['radiomics'], data['clinical']
        image_features, _ = self.encode_image(image)
        bbox_features, _ = self.encode_bbox(bbox)
        radiomics_features, _ = self.encode_radiomics(radiomics.unsqueeze(dim=-1))
        clinical_features, _ = self.encode_clinical(clinical.unsqueeze(dim=-1))
        # print(radiomics_features.shape, clinical_features.shape)
        # print('#'*20)
        # print(image_features, '\n', bbox_features, '\n',radiomics_features, '\n', clinical_features)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        bbox_features = bbox_features / bbox_features.norm(dim=1, keepdim=True)
        radiomics_features = radiomics_features / radiomics_features.norm(dim=1, keepdim=True)
        clinical_features = clinical_features / clinical_features.norm(dim=1, keepdim=True)
        image_features = image_features.mean(dim=1, keepdim=False)
        bbox_features = bbox_features.mean(dim=1, keepdim=False)
        radiomics_features = radiomics_features.mean(dim=1, keepdim=False)
        clinical_features = clinical_features.mean(dim=1, keepdim=False)


        # cosine similarity as logits
        # Calculate cosine similarity across different modalities
        logit_scale = self.logit_scale.exp()
        logits_image_bbox = logit_scale * image_features @ bbox_features.t()
        logits_image_radiomics = logit_scale * image_features @ radiomics_features.t()
        logits_image_clinical = logit_scale * image_features @ clinical_features.t()

        # shape = [global_batch_size, global_batch_size]
        logits = [[logits_image_bbox, logits_image_bbox.t()],
                             [logits_image_radiomics, logits_image_radiomics.t()],
                             [logits_image_clinical, logits_image_clinical.t()]]
        feats = [image_features, bbox_features, radiomics_features, clinical_features]
        return logits, feats

class CLIP3(CLIP2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, data):
        image, bbox, radiomics, clinical = data['image'], data['bbox'], data['radiomics'], data['clinical']
        image_features, _ = self.encode_image(image)
        bbox_features, _ = self.encode_bbox(bbox)
        radiomics_features, _ = self.encode_radiomics(radiomics.unsqueeze(dim=-1))
        clinical_features, _ = self.encode_clinical(clinical.unsqueeze(dim=-1))
        feats = torch.cat([image_features, bbox_features, radiomics_features, clinical_features], dim=1)
        # print(radiomics_features.shape, clinical_features.shape)
        # print('#'*20)
        # print(image_features, '\n', bbox_features, '\n',radiomics_features, '\n', clinical_features)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        bbox_features = bbox_features / bbox_features.norm(dim=1, keepdim=True)
        radiomics_features = radiomics_features / radiomics_features.norm(dim=1, keepdim=True)
        clinical_features = clinical_features / clinical_features.norm(dim=1, keepdim=True)
        image_features = image_features.mean(dim=1, keepdim=False)
        bbox_features = bbox_features.mean(dim=1, keepdim=False)
        radiomics_features = radiomics_features.mean(dim=1, keepdim=False)
        clinical_features = clinical_features.mean(dim=1, keepdim=False)


        # cosine similarity as logits
        # Calculate cosine similarity across different modalities
        logit_scale = self.logit_scale.exp()
        # logits_image_bbox = logit_scale * image_features @ bbox_features.t()
        logits_image_radiomics = logit_scale * image_features @ radiomics_features.t()
        logits_image_clinical = logit_scale * image_features @ clinical_features.t()

        # shape = [global_batch_size, global_batch_size]
        logits = [#[logits_image_bbox, logits_image_bbox.t()],
                             [logits_image_radiomics, logits_image_radiomics.t()],
                             [logits_image_clinical, logits_image_clinical.t()]]
        # feats = [image_features, bbox_features, radiomics_features, clinical_features]
        return logits, feats

class CLIP4(CLIP2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        resnet = generate_model(18)
        layers = list(resnet.children())
        self.proj_image = nn.Linear(512, 768)
        self.encode_image = nn.Sequential(*layers[:-1])
        # self.encode_image =
        # out = part_net(img)

    def forward(self, data):
        image, bbox, radiomics, clinical = data['image'], data['bbox'], data['radiomics'], data['clinical']
        image_features = self.encode_image(image)
        image_features = image_features.squeeze(dim=-1).squeeze(dim=-1).permute(0, 2, 1)
        image_features = self.proj_image(image_features)
        # print(image_features.shape)

        bbox_features, _ = self.encode_bbox(bbox)
        radiomics_features, _ = self.encode_radiomics(radiomics.unsqueeze(dim=-1))
        clinical_features, _ = self.encode_clinical(clinical.unsqueeze(dim=-1))
        feats = torch.cat([image_features, bbox_features, radiomics_features, clinical_features], dim=1)
        # print(radiomics_features.shape, clinical_features.shape)
        # print('#'*20)
        # print(image_features, '\n', bbox_features, '\n',radiomics_features, '\n', clinical_features)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        bbox_features = bbox_features / bbox_features.norm(dim=1, keepdim=True)
        radiomics_features = radiomics_features / radiomics_features.norm(dim=1, keepdim=True)
        clinical_features = clinical_features / clinical_features.norm(dim=1, keepdim=True)
        image_features = image_features.mean(dim=1, keepdim=False)
        bbox_features = bbox_features.mean(dim=1, keepdim=False)
        radiomics_features = radiomics_features.mean(dim=1, keepdim=False)
        clinical_features = clinical_features.mean(dim=1, keepdim=False)

        # cosine similarity as logits
        # Calculate cosine similarity across different modalities
        logit_scale = self.logit_scale.exp()
        # logits_image_bbox = logit_scale * image_features @ bbox_features.t()
        logits_image_radiomics = logit_scale * image_features @ radiomics_features.t()
        logits_image_clinical = logit_scale * image_features @ clinical_features.t()

        # shape = [global_batch_size, global_batch_size]
        logits = [#[logits_image_bbox, logits_image_bbox.t()],
                  [logits_image_radiomics, logits_image_radiomics.t()],
                  [logits_image_clinical, logits_image_clinical.t()]]
        # feats = [image_features, bbox_features, radiomics_features, clinical_features]
        return logits, feats

class CLIP_C3_V1(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 hidden_size: 768,
                 # vision
                 img_encode_type: 'vit',
                 img_size: Tuple[int, int, int],
                 vision_patch_size: 8,
                 # text
                 context_length: int,
                 transformer_heads: 12,
                 transformer_layers: 6
                 ):
        super().__init__()
        self.img_encode_type = img_encode_type
        if img_encode_type != 'vit':
            resnet = generate_model(18)
            layers = list(resnet.children())
            self.proj_image = nn.Linear(512, hidden_size)
            self.encode_image = nn.Sequential(*layers[:-1])
        else:
            self.encode_image = ViT(
                in_channels=1,
                img_size=img_size,
                patch_size=vision_patch_size,
                hidden_size=hidden_size,
                mlp_dim=3072,
                num_layers=transformer_layers,
                num_heads=transformer_heads)
        self.context_length = context_length
        self.encode_bbox = TextTransformer(
                            embed_dim=embed_dim,
                            context_length=context_length,
                            vocab_size=1024,
                            transformer_width=hidden_size,
                            transformer_heads=transformer_heads,
                            transformer_layers=transformer_layers)
        self.encode_radiomics = TabularTransformer(
                                            embed_dim=embed_dim,
                                            context_length=107,
                                            vocab_size=107,
                                            transformer_width=768,
                                            transformer_heads=12,
                                            transformer_layers=6)
        self.encode_clinical = TabularTransformer(
                                            embed_dim=32,
                                            context_length=27,
                                            vocab_size=27,
                                            transformer_width=768,
                                            transformer_heads=12,
                                            transformer_layers=6)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, data):
        image, bbox, clinical = data['image'], data['bbox'], data['clinical']
        if self.img_encode_type == 'vit':
            image_features, _ = self.encode_image(image)
            # image_features = image_features.squeeze(dim=-1).squeeze(dim=-1).permute(0, 2, 1)
            # print(image_features.shape)
        else:
            image_features = self.encode_image(image)
            image_features = image_features.squeeze(dim=-1).squeeze(dim=-1).permute(0, 2, 1)
            # print(image_features.shape)
            image_features = self.proj_image(image_features)
        # print(image_features.shape)

        # image_features = self.proj_image(image_features)
        # print(image_features.shape)
        bbox_features, _ = self.encode_bbox(bbox)
        # print(bbox_features.shape)
        # radiomics_features, _ = self.encode_radiomics(radiomics.unsqueeze(dim=-1))
        clinical_features, _ = self.encode_clinical(clinical.unsqueeze(dim=-1))
        feats = torch.cat([image_features, bbox_features, clinical_features], dim=1)
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

if __name__ == '__main__':
    # module = ViT(
    #     in_channels=1,
    #     img_size=(128, 128, 32),
    #     patch_size=8,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_layers=6,
    #     num_heads=12)

    # model = ViT
    img = torch.randn(1, 1, 32, 32, 32)
    # v1, v2 = module(img)
    # print(v1.shape, v2.shape)

    # module = TextTransformer(
    #     embed_dim=32,
    #     context_length=6,
    #     vocab_size=1000,
    #     transformer_width=768,
    #     transformer_heads=12,
    #     transformer_layers=6
    # )
    # bbox = torch.tensor([[64, 64, 64, 23, 22, 21]])
    # print(text.shape)
    # x1, x2 = module(bbox)
    # print(x1.shape, x2.shape)

    # module = RadiomicsTransformer(
    #     num_features=768,
    #     num_layers=6,
    #     num_heads=12
    # )
    cli = torch.randn(2, 8, 1)
    module = TabularEncoder(
        context_length=8,
    )
    module(cli)
    # radiomics = torch.randn(1, 107, 1)
    # r = module(radiomics)
    # print(r[0].shape, r[1].shape)
    # resnet = generate_model(18)
    # layers = list(resnet.children())
    # part_net = nn.Sequential(*layers[:-1])
    # out = part_net(img)
    # out = out.squeeze(dim=-1).squeeze(dim=-1).permute(0, 2, 1)
    # print(out.shape)
    # clinicals = torch.randn(1, 27)
    # c = module(clinicals)
    # print(c.shape)
    # print('#'*10)
    # data = {'image': img, 'bbox': bbox, 'clinical': clinicals, 'radiomics': radiomics}
    # net = CLIP4(embed_dim=32, hidden_size=768, img_size=(128, 128, 32), vision_patch_size=8, context_length=6, transformer_heads=12, transformer_layers=6)
    # net(data)
    # cosine_similarity, feats = net(data)
    # for lay in cosine_similarity:
    #     print(lay)
    # for lay in feats:
    #     print(lay.shape)



