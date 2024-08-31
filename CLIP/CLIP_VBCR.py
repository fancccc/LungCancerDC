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
        self.encode_radiomics = MLP(input_dim=107, hidden_dim=1024, num_layers=3, output_dim=hidden_size)
        self.encode_clinical = MLP(input_dim=27, hidden_dim=1024, num_layers=3, output_dim=hidden_size)
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
from typing import Sequence, Union
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

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

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
            if self.pos_embed == "perceptron" and m % p != 0:
                raise ValueError("patch_size should be divisible by img_size for perceptron.")
        self.n_patches = np.prod([im_d // p_d for im_d, p_d in zip(img_size, patch_size)]) + 1  # +1 for EHR
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

        clin_var = self.EHR_proj(clin_var)
        clin_var = clin_var.unsqueeze(dim=1)

        if self.pos_embed == "conv":
            x = x.flatten(2).transpose(-1, -2)

        x = torch.cat([clin_var, x], dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
class CLIP_C3_V2(nn.Module):
    """
    ct + bbox use tmss vit
    clinical use transformer encoder
    """
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
    module = ViT(
        in_channels=1,
        img_size=(128, 128, 32),
        patch_size=8,
        hidden_size=768,
        mlp_dim=3072,
        num_layers=6,
        num_heads=12)

    # model = ViT
    img = torch.randn(1, 1, 32, 32, 32)
    # v1, v2 = module(img)
    # print(v1.shape, v2.shape)

    module = TextTransformer(
        embed_dim=32,
        context_length=6,
        vocab_size=1000,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=6
    )
    bbox = torch.tensor([[64, 64, 64, 23, 22, 21]])
    # print(text.shape)
    # x1, x2 = module(bbox)
    # print(x1.shape, x2.shape)

    # module = RadiomicsTransformer(
    #     num_features=768,
    #     num_layers=6,
    #     num_heads=12
    # )
    module = TabularTransformer(
        embed_dim=32,
        context_length=107,
        vocab_size=107,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=6
    )
    radiomics = torch.randn(1, 107, 1)
    r = module(radiomics)
    print(r[0].shape, r[1].shape)
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



