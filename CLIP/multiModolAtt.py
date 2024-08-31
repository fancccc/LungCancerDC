# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

# from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
# from monai.utils import deprecated_arg
# import torch.nn.functional as F

# __all__ = ["ViT"]


class MultiModalAtt(nn.Module):

    def __init__(
        self,
        # in_channels: int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        output_dim = 32,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:

        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        scale = hidden_size ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(hidden_size, output_dim))

    def forward(self, x):
        # x = self.patch_embedding(x)
        # if hasattr(self, "cls_token"):
        #     cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        #     x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        # if hasattr(self, "classification_head"):
        #     x = self.classification_head(x[:, 0])

        return x, hidden_states_out

if __name__ == "__main__":
    feats = torch.randn(1, 32, 768)
    module = MultiModalAtt()
    out, h = module(feats)
    print(out.shape)