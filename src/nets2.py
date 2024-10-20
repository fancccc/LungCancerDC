# -*- coding: utf-8 -*-
'''
@file: nets2.py
@author: fanc
@time: 2024/6/11 17:08
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ViT
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
try:
    from nets import Attention
except:
    from .nets import Attention
class BboxPrompt(nn.Module):
    def __init__(self, embedding_dim=32):
        super(BboxPrompt, self).__init__()
        self.embedding = nn.Linear(1, embedding_dim)

    def forward(self, bbox):
        return self.embedding(bbox)

class CTEncoding(nn.Module):
    def __init__(self, img_size=(128, 128, 32)):
        super(CTEncoding, self).__init__()
        self.vit = ViT(
            in_channels=1,
            img_size=img_size,
            patch_size=16,
            num_heads=12,
            num_layers=12,
            hidden_size=768
        )
    def forward(self, x):
        _ , hidden_states_out = self.vit(x)
        return _, hidden_states_out

class BCNet(nn.Module):
    def __init__(self, num_classes=4):
        super(BCNet, self).__init__()
        self.bbox_encoder = BboxPrompt(embedding_dim=128)
        self.ct_encoder = CTEncoding()
        self.att1 = Attention(feat_dim=128, hidden_dim=128)
        self.att2 = Attention(feat_dim=128, hidden_dim=128)
        self.att3 = Attention(feat_dim=128, hidden_dim=128)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classification = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def prompt_attention(self, bbox_feat, hidden_states_out, n):
        bbox_len = bbox_feat.shape[1]
        vit_layer = hidden_states_out[n]
        att_feat = []
        for i in range(bbox_len):
            if n == -1:
                att_feat.append(self.att1(bbox_feat[:, i:i+1, ...], vit_layer.transpose(2, 1)))
            elif n == -5:
                att_feat.append(self.att2(bbox_feat[:, i:i + 1, ...], vit_layer.transpose(2, 1)))
            elif n == -9:
                att_feat.append(self.att3(bbox_feat[:, i:i + 1, ...], vit_layer.transpose(2, 1)))
        return torch.cat(att_feat, dim=1)

    def forward(self, img, bbox):
        _, hidden_states_out = self.ct_encoder(img)
        bbox_feat = self.bbox_encoder(bbox)
        # print(hidden_states_out[-1].shape, bbox_feat[:, 0:1, ...].shape)
        att_feat1 = self.prompt_attention(bbox_feat, hidden_states_out, -1)
        att_feat2 = self.prompt_attention(bbox_feat, hidden_states_out, -5)
        att_feat3 = self.prompt_attention(bbox_feat, hidden_states_out, -9)
        att_feat = torch.cat((att_feat1, att_feat2, att_feat3), dim=-1)
        x = att_feat.permute(0, 2, 1)
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        x = self.classification(x)

        return x


class EHREmbedding(nn.Module):
    def __init__(self, max_len=50, value_dim=1, embedding_dim=256):
        super(EHREmbedding, self).__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.position_embedding = nn.Parameter(torch.rand(1, max_len, embedding_dim))
        nn.init.uniform_(self.position_embedding, -0.1, 0.1)
        self.fc = nn.Linear(value_dim, embedding_dim)
    def forward(self, values):
        mask = torch.isnan(values)
        mean, std = 0, 0.1  # Example mean and standard deviation for the normal distribution
        values = torch.where(mask, torch.normal(mean, std, size=values.shape, device=values.device), values)
        # values = torch.where(mask, torch.zeros_like(values), values)
        transformed_values = self.fc(values) # B, 47, 768
        pos_embeddings = self.position_embedding[:, :values.size(1), :] # B, 47, 768
        valid_pos_embeddings = torch.where(mask, torch.zeros_like(pos_embeddings), pos_embeddings)
        encoded_values = transformed_values + valid_pos_embeddings
        combined = encoded_values * (1 - mask.float())
        combined = combined.mean(dim=1)
        # print(encoded_values.shape, combined.shape, 1 - mask.float(), combined)
        return combined

class BCENet(nn.Module):
    def __init__(self, num_classes=4):
        super(BCENet, self).__init__()
        self.bbox_encoder = BboxPrompt(embedding_dim=128)
        self.ct_encoder = CTEncoding()
        self.att1 = Attention(feat_dim=128, hidden_dim=128)
        self.att2 = Attention(feat_dim=128, hidden_dim=128)
        self.att3 = Attention(feat_dim=128, hidden_dim=128)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classification = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.ehr_embedding = EHREmbedding(max_len=50, value_dim=1, embedding_dim=128)
        # self.deconv1 = nn.ConvTranspose1d(768, 256, kernel_size=3, stride=1, padding=1)
        # self.deconv2 = nn.ConvTranspose1d(256, 32, kernel_size=3, stride=1, padding=1)  # intermediate size adjustment
        # self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # expand to (128, 128)
        # self.deconv4 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # ensure size matches exactly

    def prompt_attention(self, key_feat, hidden_states_out, n):
        bbox_len = key_feat.shape[1]
        vit_layer = hidden_states_out[n]
        att_feat = []
        for i in range(bbox_len):
            if n == -1:
                att_feat.append(self.att1(key_feat[:, i:i+1, ...], vit_layer.transpose(2, 1)))
            elif n == -5:
                att_feat.append(self.att2(key_feat[:, i:i + 1, ...], vit_layer.transpose(2, 1)))
            elif n == -9:
                att_feat.append(self.att3(key_feat[:, i:i + 1, ...], vit_layer.transpose(2, 1)))
        return torch.cat(att_feat, dim=1)

    def forward(self, img, bbox, EHR):
        EHR_Encoded = self.ehr_embedding(EHR) # b, 1, 128
        # print(EHR_Encoded.shape)
        _, hidden_states_out = self.ct_encoder(img)
        bbox_feat = self.bbox_encoder(bbox) # b, 6, 128
        key_feat = torch.cat((bbox_feat, EHR_Encoded.unsqueeze(1)), dim=1)
        # print(hidden_states_out[-1].shape, bbox_feat[:, 0:1, ...].shape)
        att_feat1 = self.prompt_attention(key_feat, hidden_states_out, -1)
        att_feat2 = self.prompt_attention(key_feat, hidden_states_out, -5)
        att_feat3 = self.prompt_attention(key_feat, hidden_states_out, -9)
        att_feat = torch.cat((att_feat1, att_feat2, att_feat3, key_feat), dim=-1) # b, 7, 512
        # print(att_feat.shape, key_feat.shape)
        x = att_feat.permute(0, 2, 1)
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        # print(x.shape)
        x = self.classification(x)

        return x

class SegEncoder(nn.Module):
    def __init__(self):
        super(SegEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)
        x = self.avg_pool(x)
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        # print(x.shape)
        return x

class SBCENet(BCENet):
    def __init__(self, num_classes=4, *args, **kwargs):
        super(SBCENet, self).__init__( *args, **kwargs)
        self.seg_encoder = SegEncoder()
        self.classification = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, bbox, EHR, seg):
        EHR_Encoded = self.ehr_embedding(EHR) # b, 1, 128
        # print(EHR_Encoded.shape)
        _, hidden_states_out = self.ct_encoder(img)
        bbox_feat = self.bbox_encoder(bbox) # b, 6, 128
        seg_feat = self.seg_encoder(seg)
        key_feat = torch.cat((bbox_feat, EHR_Encoded.unsqueeze(1), seg_feat), dim=1)
        # print(hidden_states_out[-1].shape, bbox_feat[:, 0:1, ...].shape)
        att_feat1 = self.prompt_attention(key_feat, hidden_states_out, -1)
        att_feat2 = self.prompt_attention(key_feat, hidden_states_out, -5)
        att_feat3 = self.prompt_attention(key_feat, hidden_states_out, -9)
        att_feat = torch.cat((att_feat1, att_feat2, att_feat3, key_feat), dim=-1) # b, 8, 512
        # print(att_feat.shape, key_feat.shape)
        x = att_feat.permute(0, 2, 1)
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        # print(x.shape)
        x = self.classification(x)
        return x


class RadiomicsTransformer(nn.Module):
    def __init__(self, num_features, num_layers, num_heads):
        super(RadiomicsTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(num_features)
        encoder_layers = TransformerEncoderLayer(d_model=num_features, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        # self.decoder = nn.Linear(num_features, 1)
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # output = self.decoder(output)
        return output

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            self.encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            self.encoding[:, 1::2] = torch.cos(position * div_term[:-1])

        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return x

from CLIP.CLIP_VBCR import TabularTransformer
class RadiomicsNet(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs):
        super(RadiomicsNet, self).__init__()
        self.radiomics_encoder = TabularTransformer(
        embed_dim=32,
        context_length=107,
        vocab_size=107,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=6
    )
        self.classification = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, data):
        x = data['radiomics'].unsqueeze(dim=-1)
        # print(x.shape)
        x, _ = self.radiomics_encoder(x)
        # print(x.shape)
        if torch.isnan(x).any():
            print('NaN in radiomics_encoder', x)
        x = self.classification(x.mean(dim=1).view(x.size(0), -1))
        return x

from multisurv.nets import ClinicalNet
class CliNet(nn.Module):
    def __init__(self, num_classes=3, clinical_length=27):
        super(CliNet, self).__init__()
        # self.cli_encoder = TabularTransformer(
        #                                     embed_dim=32,
        #                                     context_length=27,
        #                                     vocab_size=27,
        #                                     transformer_width=768,
        #                                     transformer_heads=12,
        #                                     transformer_layers=6)
        self.cli_encoder = ClinicalNet(output_vector_size=512, embedding_dims=clinical_length)
        self.classification = nn.Sequential(
                                nn.Linear(512, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(256, num_classes))
    def forward(self, data):
        x = data['clinical']#.unsqueeze(dim=-1)
        # print(x.shape)
        x = self.cli_encoder(x)
        # print(x.shape)
        if torch.isnan(x).any():
            print('NaN in radiomics_encoder', x)
        x = self.classification(x)
        return x
class MCTAtt(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(MCTAtt, self).__init__()
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(feat_dim, hidden_dim)
        self.key = nn.Linear(feat_dim, hidden_dim)
        self.value = nn.Linear(feat_dim, hidden_dim)
    def forward(self, query, memory):
        # batch_size, c1, seq_len = query.size()
        # c2 = memory.size(1)
        q = self.query(query)
        k = self.key(memory)
        v = self.value(memory)
        # print(q.shape, k.shape, v.shape)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.hidden_dim ** 0.5
        # attn_weights = attn_weights.mean(dim=1).view(batch_size, c2, 1)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attended = torch.matmul(attn_weights, v)
        # print(attn_weights.shape, v.shape, attended.shape)
        # attended = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, c, mem_len)
        # print(attn_weights.shape, attended.shape)
        return attended

class MCREBSNet(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs):
        super(MCREBSNet, self).__init__()
        self.vit128 = ViT(
            in_channels=1,
            img_size=(128, 128, 32),
            patch_size=16,
            num_heads=12,
            num_layers=12,
            hidden_size=768)
        self.vit32 = ViT(
            in_channels=1,
            img_size=(32, 32, 32),
            patch_size=16,
            num_heads=12,
            num_layers=12,
            hidden_size=768)
        self.radiomics_encoder = RadiomicsTransformer(num_features=128, num_layers=2, num_heads=8)
        self.ehr_embedding = EHREmbedding(max_len=50, value_dim=1, embedding_dim=128)
        self.bbox_encoder = BboxPrompt(embedding_dim=128)
        self.seg_encoder = SegEncoder()
        self.mct1 = MCTAtt(feat_dim=768, hidden_dim=768)
        self.mct2 = MCTAtt(feat_dim=768, hidden_dim=768)
        self.mct3 = MCTAtt(feat_dim=768, hidden_dim=768)
        self.radiomics_att = MCTAtt(feat_dim=128, hidden_dim=128)
        self.trans_conv1 = nn.ConvTranspose1d(in_channels=768, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trans_conv2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.prompt_att_ehr = MCTAtt(feat_dim=128, hidden_dim=128)
        self.prompt_att_seg = MCTAtt(feat_dim=128, hidden_dim=128)
        self.prompt_att_bbox = MCTAtt(feat_dim=128, hidden_dim=128)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classification = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, data):
        ct32 = data['ct32']
        ct128 = data['ct128']
        radiomic = data['radiomic']
        ehr = data['clinical']
        bbox = data['bbox']
        seg = data['seg']
        _32, hidden_states_out32 = self.vit32(ct32)
        _128, hidden_states_out128 = self.vit128(ct128)
        radiomic = self.radiomics_encoder(radiomic)
        EHR_Encoded = self.ehr_embedding(ehr)
        bbox_feat = self.bbox_encoder(bbox) # b, 6, 128
        seg_feat = self.seg_encoder(seg)
        mct1 = self.mct1(_32, hidden_states_out128[-1]) + _32
        mct2 = self.mct1(_32, hidden_states_out128[-5]) + _32
        mct3 = self.mct1(_32, hidden_states_out128[-9]) + _32
        mct = torch.cat([mct1, mct2, mct3], dim=1)
        mct = self.trans_conv1(mct.permute(0, 2, 1))
        # print(mct.shape)
        mct = self.trans_conv2(mct)
        radiomic = self.radiomics_att(radiomic, radiomic)
        values = torch.cat([mct.permute(0, 2, 1), radiomic], dim=1)
        ehr_att = self.prompt_att_ehr(EHR_Encoded.unsqueeze(1), values) + EHR_Encoded
        bbox_att = self.prompt_att_bbox(bbox_feat, values) + bbox_feat
        seg_att = self.prompt_att_seg(seg_feat, values) + seg_feat
        prompt = torch.cat([ehr_att, bbox_att, seg_att], dim=1)
        out = torch.cat([prompt, values], dim=1)
        # print(out.shape)


        # print(att_feat.shape, key_feat.shape)
        out = out.permute(0, 2, 1)
        out = self.avg_pool(out)
        out = self.classification(out.view(out.size(0), -1))
        # print(out.shape)
        # print(_32.shape, _128.shape, radiomic.shape, EHR_Encoded.shape, bbox_feat.shape, seg_feat.shape)
        return out

    def prompt_attention(self, key_feat, values):
        bbox_len = key_feat.shape[1]
        vit_layer = values
        att_feat = []
        for i in range(bbox_len):
            att_feat.append(self.prompt_att(key_feat[:, i:i + 1, ...], vit_layer.transpose(2, 1)))
        return torch.cat(att_feat, dim=1)

class ViTBase(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs):
        super(ViTBase, self).__init__()
        self.vit128 = ViT(
            in_channels=1,
            img_size=(128, 128, 32),
            patch_size=16,
            num_heads=12,
            num_layers=12,
            hidden_size=768)
        self.vit32 = ViT(
            in_channels=1,
            img_size=(32, 32, 32),
            patch_size=16,
            num_heads=12,
            num_layers=12,
            hidden_size=768)
        self.radiomics_encoder = RadiomicsTransformer(num_features=128, num_layers=2, num_heads=8)
        self.ehr_embedding = EHREmbedding(max_len=50, value_dim=1, embedding_dim=128)
        self.bbox_encoder = BboxPrompt(embedding_dim=128)
        self.seg_encoder = SegEncoder()
        self.mct1 = MultiHeadAttention(feat_dim=768, hidden_dim=768)
        self.mct2 = MultiHeadAttention(feat_dim=768, hidden_dim=768)
        self.mct3 = MultiHeadAttention(feat_dim=768, hidden_dim=768)
        self.mct4 = MultiHeadAttention(feat_dim=768, hidden_dim=768)
        self.prompt_att = ViT( in_channels=115,
                                img_size=(4, 4, 8),
                                patch_size=2,
                                num_heads=12,
                                num_layers=12,
                                hidden_size=768)
        self.ca = ChannelAttention(num_channels=56, reduction_ratio=16)
        self.classification = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, data):
        ct32 = data['ct32']
        ct128 = data['ct128']
        radiomic = data['radiomic']
        ehr = data['clinical']
        bbox = data['bbox']
        seg = data['seg']
        _32, hidden_states_out32 = self.vit32(ct32)
        _128, hidden_states_out128 = self.vit128(ct128)
        radiomic = self.radiomics_encoder(radiomic)
        EHR_Encoded = self.ehr_embedding(ehr).unsqueeze(1)
        bbox_feat = self.bbox_encoder(bbox) # b, 6, 128
        seg_feat = self.seg_encoder(seg)
        mct1 = self.mct1(_32, hidden_states_out128[3-1], hidden_states_out128[3-1])
        mct2 = self.mct2(_32, hidden_states_out128[6-1], hidden_states_out128[6-1])
        mct3 = self.mct3(_32, hidden_states_out128[9-1], hidden_states_out128[9-1])
        mct4 = self.mct4(_32, hidden_states_out128[12-1], hidden_states_out128[12-1])
        mct = torch.cat([mct1, mct2, mct3, mct4, _32], dim=1)
        # print(mct.shape)
        prompt = torch.cat([radiomic, EHR_Encoded, bbox_feat, seg_feat], dim=1)
        prompt = prompt.view(prompt.shape[0], prompt.shape[1], 4, 4, 8)
        # print(prompt.shape)
        p, hidden_states_out = self.prompt_att(prompt)
        # print(p.shape)
        out = torch.cat([p, mct], dim=1)
        # print(out.shape)
        out = self.ca(out)
        out = self.classification(out.mean(dim=1).view(out.size(0), -1))
        # print(prompt.shape, prompt.transpose(2, 1).shape)
        l12 = torch.norm(torch.bmm(_128, _32.transpose(2, 1)), p=1) / (torch.norm(_128, p=2) * torch.norm(_32, p=2))
        l23 = torch.norm(torch.bmm(_32, p.transpose(2, 1)), p=1) / (torch.norm(_32, p=2) * torch.norm(p, p=2))
        l13 = torch.norm(torch.bmm(_128, p.transpose(2, 1)), p=1) / (torch.norm(_128, p=2) * torch.norm(p, p=2))
        l123 = 0.33*l12 + 0.33*l23 + 0.33*l13
        # print(l123)
        return out, l123

class MultiHeadAttention(nn.Module):
    def __init__(self, feat_dim, hidden_dim, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.feat_dim = feat_dim
        self.query = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.key = nn.Linear(feat_dim, hidden_dim, bias=False)
        self.value = nn.Linear(feat_dim, hidden_dim, bias=False)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        query = self.query(query).view(batch_size, query_len, self.heads, self.feat_dim // self.heads).transpose(1, 2)
        key = self.key(key).view(batch_size, key_len, self.heads, self.feat_dim // self.heads).transpose(1, 2)
        value = self.value(value).view(batch_size, value_len, self.heads, self.feat_dim // self.heads).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.feat_dim)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, query_len, self.feat_dim)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels)

    def forward(self, x):
        squeeze = torch.mean(x, dim=-1, keepdim=True)
        excitation = F.relu(self.fc1(squeeze.squeeze(-1)))
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = excitation.view(-1, self.num_channels, 1)
        scale = x * excitation
        return scale


if __name__ == '__main__':
    bbox = torch.tensor([0.1, 0.3, 0.2, 0.3, 0.2, 0.1]).unsqueeze(0).unsqueeze(-1)
    EHR = torch.randn(1, 27)#.unsqueeze(-1)
    # net = BboxPrompt()
    # print(EHR.shape)
    radiomic = torch.randn(1, 107, 1)
    # model = RadiomicsNet(2)
    # out = model(radiomic)
    # print(out.shape)

    ct128 = torch.randn(1, 1, 128, 128, 32)
    ct32 = torch.randn(1, 1, 32, 32, 32)
    seg = torch.randn(1, 1, 128, 128, 32)
    data = {'ct32' : ct32, 'ct128' : ct128, 'radiomics' : radiomic, 'clinical' : EHR, 'bbox' : bbox, 'seg' : seg}
    net = CliNet()
    print(net(data).shape)
    # # net = BCENet()
    # # out = net(img, bbox, EHR)
    # # print(out.shape)
    # # net = SegEncoder()
    # # out = net(img)
    # net = SBCENet()
    # out = net(img, bbox, EHR, img)
    # print(out.shape)
    # net = RadiomicsNet()
    # out = net(data)
    # ma = MultiHeadAttention(768, 768)
    # m1 = torch.randn(1, 8, 768)
    # m2 = torch.randn(1, 128, 768)
    # x = ma(m1, m2, m2)
    # print(out.shape)

