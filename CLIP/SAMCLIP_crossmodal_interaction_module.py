from .mask_decoder import SAM_mask_decoder
from .prompt_encoder import SAM_prompt_encoder
from .CLIP_encoder import CLIP
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

class SAMCLIP(nn.Module):
    def __init__(self,
                 image_size: int,
                 text_size: int,
                 mask_size: int,
                 box_size: int
                 ):
        super().__init__()

        self.image = image_size
        self.text = text_size
        self.mask = mask_size
        self.box = box_size

        self.Conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
        )

        self.CLIP = CLIP(16, 16, [[2, 2, 2, 2], 8], image_size, 16, text_size, 4, 16, 16, 4)

        self.pmt_encoder = SAM_prompt_encoder(16, 16, [mask_size, box_size], 8)
        self.mk_decoder = SAM_mask_decoder(16, [image_size], 32, 256)

        self.attention = Attention_Fusion(16, 1024, 128, 0, 256)

    def forward(self, image, text, mask, box):

        image_bedding, text_bedding = self.CLIP(image, text)
        att = self.attention(image_bedding, text_bedding)
        att_con = torch.cat([att, image_bedding], dim=1)

        conv = self.Conv(att_con)
        img_con = torch.cat([conv, image], dim=1)
        points = random_points_in_mask(mask, 10)
        pmt = self.pmt_encoder([points[0:], points[1:]], box, [])
        pm = self.mk_decoder(img_con, 1024, pmt)
        return pm



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

class Attention_Fusion(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob, output_size ):
        """
        Multi_Head_Self_Attention_Fusion (MHSAF) Module and a 3×3×3 Conv
        """
        super(Attention_Fusion, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.conv = conv3x3x3(hidden_size, output_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, image, text):
        mixed_query_layer = self.query(image)
        mixed_key_layer = self.key(text)
        mixed_value_layer = self.value(text)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + mixed_query_layer)

        return self.conv(hidden_states)


def random_points_in_mask(mask, num_points):

    mask = mask.astype(bool)
    indices = np.argwhere(mask)
    selected_indices = np.random.choice(len(indices), num_points, replace=False)
    selected_points = indices[selected_indices]

    return selected_points