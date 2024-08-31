import torch
import torch.nn as nn
import torch.nn.functional as F
# from funasr.models.ct_transformer.model import CTTransformer
from src.resnet import conv1x1x1, get_inplanes, conv3x3x3, BasicBlock, Bottleneck
from functools import partial
from typing import Tuple, Union
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
from torchvision.models import resnet101, resnet18, resnet34, resnet50
class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()
        # if n_classes == 2:
        #     n_classes = 1
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, 7, 7),
                               stride=(1, 1, 1),
                               padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x
def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

class KeyValueEncoding(nn.Module):
    def __init__(self, num_keys=47, value_dim=1, embedding_dim=128, output_dim=128):
        super(KeyValueEncoding, self).__init__()
        self.key_embedding = nn.Embedding(num_keys, embedding_dim)  # +1 for padding token
        self.fc_value = nn.Linear(value_dim, embedding_dim)
        self.fc_combined = nn.Linear(embedding_dim * 2, output_dim)
    def forward(self, keys, values):
        keys = torch.tensor(keys, dtype=torch.long)
        mask = torch.isnan(values).float()
        embedded_keys = self.key_embedding(keys)
        values = torch.where(torch.isnan(values), torch.zeros_like(values), values)
        processed_values = self.fc_value(values.unsqueeze(-1))
        combined = torch.cat((embedded_keys, processed_values), dim=-1)
        combined = combined * mask.unsqueeze(-1)  # 应用mask
        combined_sum = torch.sum(combined, dim=1)
        mask_sum = torch.sum(mask, dim=1, keepdim=True)
        combined = combined_sum / (mask_sum + 1e-5)  # 求平均，避免除以0
        combined = F.relu(self.fc_combined(combined))
        return combined
class EHRTransformer(nn.Module):
    def __init__(self,input_dim=128, d_model=256, n_head=8, n_layers_feat=1,
                 n_layers_shared=1, n_layers_distinct=1,
                 dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_feat = nn.TransformerEncoder(layer, num_layers=n_layers_feat)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_shared = nn.TransformerEncoder(layer, num_layers=n_layers_shared)
        # layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        # self.model_distinct = nn.TransformerEncoder(layer, num_layers=n_layers_distinct)
    def forward(self, x):
        x = self.fc(x)
        feat = self.model_feat(x)
        h_shared = self.model_shared(feat)
        # h_distinct = self.model_distinct(feat)
        return h_shared #, h_distinct

class CTEncoder(nn.Module):
    def __init__(self, model_depth=18, feat_dim=512, hidden_dim=256):
        super(CTEncoder, self).__init__()
        resnet = generate_model(model_depth=model_depth)
        self.model_feat = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
        )
        self.model_shared = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        # self.model_shared.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=hidden_dim)

        resnet = generate_model(model_depth=model_depth)
        self.model_spec = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        # self.model_spec.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=hidden_dim)

        self.shared_project = nn.Sequential(
            nn.Linear(resnet.fc.in_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.spec_project = nn.Sequential(
            nn.Linear(resnet.fc.in_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        feature = self.model_feat(x)
        shared = self.model_shared(feature)
        spce = self.model_spec(feature)
        shared = self.shared_project(shared.view(x.size(0), -1))
        spec = self.spec_project(spce.view(x.size(0), -1))
        return shared, spec

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_length, _ = query.size()

        # Linear projections
        query = self.query_proj(query).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.out_proj(attn_output)

        return output

class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * 256, 256)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 6)
        # self.ct_dims = torch.tensor([ct_dims[-3], ct_dims[-2], ct_dims[-1]], device=device).float().repeat(2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MultiModalMultiHeadAttention(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.attention_mod1 = MultiHeadAttention(feat_dim, hidden_dim)
        self.attention_mod2 = MultiHeadAttention(feat_dim, hidden_dim)

    def forward(self, mod1_data, mod2_data):
        # mod1_data 和 mod2_data 是两个不同模态的输入数据
        # 模态1使用模态2的数据计算注意力
        mod1_attended = self.attention_mod1(mod1_data, mod2_data)
        # 模态2使用模态1的数据计算注意力
        mod2_attended = self.attention_mod2(mod2_data, mod1_data)
        # 融合两个模态的输出
        # fused_output_1 = mod1_attended * 0.2 + mod2_attended * 0.8 # 这里可以是加法、拼接等其他方式
        # fused_output_2 = mod1_attended * 0.8 + mod2_attended * 0.2
        return torch.cat([mod1_attended, mod2_attended], dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(feat_dim, hidden_dim)
        self.key = nn.Linear(feat_dim, hidden_dim)
        self.value = nn.Linear(feat_dim, hidden_dim)
        self.num_heads = 8
        self.head_dim = hidden_dim // self.num_heads

    def forward(self, x, memory):
        # x是查询数据，memory是键和值的数据
        batch_size, seq_len = x.size()
        _, mem_len = memory.size()

        q = self.query(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(memory).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(memory).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attended = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return attended
class Attention(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(feat_dim, hidden_dim)
        self.key = nn.Linear(feat_dim, hidden_dim)
        self.value = nn.Linear(feat_dim, hidden_dim)

    def forward(self, x, memory):
        # x是查询数据，memory是键和值的数据
        batch_size, c, seq_len = x.size()
        _, c, mem_len = memory.size()

        q = self.query(x)
        k = self.key(memory)
        v = self.value(memory)
        # print(q.shape, k.shape, v.shape)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.hidden_dim ** 0.5
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attended = torch.matmul(attn_weights, v)
        # attended = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, c, mem_len)
        # print(attn_weights.shape, attended.shape)
        return attended
class DCNet(nn.Module):
    def __init__(self, num_classes=4,
                 embedding_dim=128,
                 hidden_dim=256,
                 model_depth=18,
                 num_keys=47, num_heads=8, crop_size=100, img_size=(256, 256, 300), device=torch.device('cuda')):
        super(DCNet, self).__init__()
        self.crop_size = crop_size
        self.enr_encoding = KeyValueEncoding(num_keys=num_keys, value_dim=1, embedding_dim=embedding_dim, output_dim=embedding_dim)
        self.ehr_transformer = EHRTransformer(input_dim=embedding_dim, d_model=hidden_dim, n_head=num_heads)
        self.ct_dispose = CTEncoder(model_depth=model_depth, feat_dim=512, hidden_dim=hidden_dim)
        self.modal_fuse = MultiModalMultiHeadAttention(feat_dim=256, hidden_dim=256)

        self.detection_header = DetectionHead()

        self.cross_attention1 = MultiHeadCrossAttention(d_model=hidden_dim, num_heads=num_heads)

        self.crop_resnet = generate_model(model_depth=model_depth)
        # self.cross_attention2 = MultiHeadCrossAttention(d_model=hidden_dim, num_heads=num_heads)

        self.classifier = nn.Linear(hidden_dim*4, num_classes)

    def crop_img(self, bbox, ct):
        b, c, d, h, w = ct.shape
        cropped_images = torch.zeros((b, c, self.crop_size, self.crop_size, self.crop_size), dtype=ct.dtype,
                                     device=ct.device)
        for i in range(b):
            center_z = bbox[i][2].detach().int()
            center_y = bbox[i][1].detach().int()
            center_x = bbox[i][0].detach().int()

            z_min = int(torch.clamp(center_z - self.crop_size // 2, min=0))
            y_min = int(torch.clamp(center_y - self.crop_size // 2, min=0))
            x_min = int(torch.clamp(center_x - self.crop_size // 2, min=0))
            z_max = int(torch.clamp(center_z + self.crop_size // 2, max=d))
            y_max = int(torch.clamp(center_y + self.crop_size // 2, max=h))
            x_max = int(torch.clamp(center_x + self.crop_size // 2, max=w))

            cropped = ct[i, :, z_min:z_max, y_min:y_max, x_min:x_max]
            # 如果裁剪区域小于指定的crop_size，则用0填充
            pad_size = (
                0, self.crop_size - (x_max - x_min),  # width padding
                0, self.crop_size - (y_max - y_min),  # height padding
                0, self.crop_size - (z_max - z_min),  # depth padding
            )
            cropped_images[i] = F.pad(cropped, pad_size, 'constant', 0)

        return cropped_images



    def forward(self, ehr, ct, only_detection=True):
        ehr = self.enr_encoding(ehr[:, 0], ehr[:, 1])
        # print('ehr shape: ', ehr.shape)
        h_ehr_share, h_ehr_distinct = self.ehr_transformer(ehr)
        # h_ehr_share, h_ehr_distinct = torch.randn(2, 256), torch.randn(2, 256)
        h_ct_share, h_ct_distinct = self.ct_dispose(ct)
        t_d, t_c = self.modal_fuse(h_ct_share, h_ehr_share) #B*256*1

        t_d = torch.cat((t_d, h_ct_distinct.unsqueeze(-1)), dim=-1).transpose(2, 1)# B*2*256
        bbox = self.detection_header(t_d)
        cls = None
        if not only_detection:
            t_c = torch.cat((t_c, h_ehr_distinct.unsqueeze(-1)), dim=-1).transpose(2, 1)
            # print(t_c.shape)
            crop_ct = self.crop_img(bbox, ct)
            t2 = self.crop_resnet(crop_ct)
            ts = torch.cat((t_c, t2.view(t2.size(0), 2, 256)), dim=1) # B*4*256
            # print(ts.view(-1).shape)
            # feat = self.cross_attention2(ts, ts, ts)
            # cls
            cls = self.classifier(ts.view(ts.size(0), -1))
        return bbox, cls

class DetectionLungCancer(nn.Module):
    def __init__(self, num_classes=4,
                 embedding_dim=128,
                 hidden_dim=256,
                 model_depth=18,
                 num_keys=47, num_heads=8):
        super(DetectionLungCancer, self).__init__()
        resnet = generate_model(model_depth=model_depth)
        self.resnet = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.conv1 = nn.Conv3d(512, 1, 5, stride=1, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.upsample1 = nn.ConvTranspose3d(
                            in_channels=512,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1)
        self.upsample2 = nn.ConvTranspose3d(
                            in_channels=256,
                            out_channels=128,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1)
        self.upsample3 = nn.ConvTranspose3d(
                            in_channels=128,
                            out_channels=1,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1)
        # self.upsample4 = nn.ConvTranspose3d(
        #                     in_channels=64,
        #                     out_channels=1,
        #                     kernel_size=3,
        #                     stride=2,
        #                     padding=1,
        #                     output_padding=1)

        # self.enr_encoding = KeyValueEncoding(num_keys=num_keys, value_dim=1, embedding_dim=embedding_dim, output_dim=embedding_dim)
        # self.ehr_transformer = EHRTransformer(input_dim=embedding_dim, d_model=hidden_dim, n_head=num_heads)
        # self.ct_dispose = CTEncoder(model_depth=model_depth, feat_dim=512, hidden_dim=hidden_dim)
        # self.modal_fuse = MultiModalMultiHeadAttention(feat_dim=256, hidden_dim=256)
        # self.detection_header = DetectionHead()
        # self.cross_attention1 = MultiHeadCrossAttention(d_model=hidden_dim, num_heads=num_heads)
        # self.crop_resnet = generate_model(model_depth=model_depth)
        # # self.cross_attention2 = MultiHeadCrossAttention(d_model=hidden_dim, num_heads=num_heads)
        # self.classifier = nn.Linear(hidden_dim*4, num_classes)


    def forward(self, ehr, ct, only_detection=True):
        # ehr = self.enr_encoding(ehr[:, 0], ehr[:, 1])
        # h_ehr_share, h_ehr_distinct = self.ehr_transformer(ehr)
        # h_ehr_share, h_ehr_distinct = torch.randn(2, 256), torch.randn(2, 256)
        feat = self.resnet(ct)
        # print(feat.shape)
        # y = feat.mean(dim=1, keepdim=True)
        # print(y.shape)
        # y = self.conv1(feat)
        # y = self.sigmoid(y)
        # feat = feat * y.expand_as(feat)
        feat = self.upsample1(feat)
        feat = self.upsample2(feat)
        feat = self.upsample3(feat)
        # t_d, t_c = self.modal_fuse(h_ct_share, h_ehr_share) #B*256*1
        # t_d = torch.cat((t_d, h_ct_distinct.unsqueeze(-1)), dim=-1).transpose(2, 1)# B*2*256
        # bbox = self.detection_header(t_d)
        # t_c = torch.cat((t_c, h_ehr_distinct.unsqueeze(-1)), dim=-1).transpose(2, 1)
        # crop_ct = self.crop_img(bbox, ct)
        # t2 = self.crop_resnet(crop_ct)
        # ts = torch.cat((t_c, t2.view(t2.size(0), 2, 256)), dim=1) # B*4*256
        # cls = self.classifier(ts.view(ts.size(0), -1))

        return feat, None

class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
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
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, ehr, x_in, y):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits, None

class EHRNet(nn.Module):
    def __init__(self,
                 num_classes=4,
                 embedding_dim=128,
                 hidden_dim=256,
                 num_keys=47, num_heads=8):
        super(EHRNet, self).__init__()
        self.enr_encoding = KeyValueEncoding(num_keys=num_keys, value_dim=1, embedding_dim=embedding_dim, output_dim=embedding_dim)
        self.ehr_transformer = EHRTransformer(input_dim=embedding_dim, d_model=hidden_dim, n_head=num_heads)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(64, num_classes))
    def forward(self, ehr):
        ehr_enc = self.enr_encoding(ehr[:, 0], ehr[:, 1])
        # print(ehr[:, 0], ehr[:, 1])
        # print(ehr_enc.shape)
        ehr_enc = self.ehr_transformer(ehr_enc)
        # print(ehr_enc.shape)
        ehr_enc = self.fc(ehr_enc)
        return ehr_enc


class KeyValuePositionEmbedding(nn.Module):
    def __init__(self, max_len=50, value_dim=1, embedding_dim=256):
        super(KeyValuePositionEmbedding, self).__init__()
        self.pe = nn.Parameter(torch.rand(1, max_len, embedding_dim))
        self.pe.data.uniform_(-0.1, 0.1)
        self.fc_k = nn.Linear(value_dim, embedding_dim)
        self.fc_v = nn.Linear(value_dim, embedding_dim)

    def forward(self, keys, values):
        temp = []
        mask = torch.isnan(values).float()
        temp.append(mask)
        values = torch.where(torch.isnan(values), torch.zeros_like(values), values)
        temp.append(values)
        # values = values / (mask.sum(dim=-1, keepdim=True) + 1)
        temp.append(values)
        values = self.fc_v(values.unsqueeze(-1))
        temp.append(values)
        keys.uniform_(-0.1, 0.1)
        keys = self.fc_k(keys.unsqueeze(-1))
        temp.append(keys)
        combined = keys + values + self.pe[:, :values.size(1)]  # B, len, embedding_ding
        temp.append(combined)
        combined = combined * (1 - mask).unsqueeze(-1)
        temp.append(combined)
        combined = combined.mean(dim=1)
        temp.append(combined)
        if torch.isnan(combined).any():
            for i, j in enumerate(temp):
                print(i, j)
            return

        return combined
class CENet(nn.Module):
    def __init__(self, num_classes, model_depth=34, hidden_dim=512, max_len=50, num_heads=8, embedding_dim=128):
        super(CENet, self).__init__()
        resnet = generate_model(model_depth)
        self.ct_encoder1 = nn.Sequential(resnet.conv1,
                                        resnet.bn1,
                                        resnet.layer1,
                                        resnet.layer2,
                                        resnet.layer3,
                                        resnet.layer4,
                                        resnet.avgpool)

        self.kv_encoder = KeyValuePositionEmbedding(max_len=max_len, embedding_dim=embedding_dim)
        self.ehr_transformer = EHRTransformer(input_dim=embedding_dim, d_model=hidden_dim, n_head=num_heads)

        resnet = generate_model(model_depth)
        self.ct_decoder2 = nn.Sequential(resnet.conv1,
                            resnet.bn1,
                            resnet.layer1,
                            resnet.layer2,
                            resnet.layer3,
                            resnet.layer4,
                            resnet.avgpool)
        self.fuse_modal = MultiModalMultiHeadAttention(feat_dim=hidden_dim, hidden_dim=hidden_dim)
        self.ca = ChannelAttention(in_channels=3)

        self.fc = nn.Sequential(nn.Linear(hidden_dim*3, 128),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128, num_classes))

        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.bn2 = nn.BatchNorm3d(num_features=3)



    def forward(self, img, ehr, img_s):
        temp = []
        f1 = self.ct_encoder1(img) # b, 512, 1, 1, 1
        temp.append(f1)
        f2 = self.kv_encoder(ehr[:, 0], ehr[:, 1]) # B, 128
        # print(f2.shape)
        temp.append(f2)
        f2 = self.ehr_transformer(f2) # b, 512
        temp.append(f2)
        feat = self.fuse_modal(f1.view(f1.size(0), -1), f2) # B, 1024, 1
        temp.append(feat)
        feat = self.bn1(feat) #B, 1024, 1
        temp.append(feat)
        # print(feat.shape)
        f3 = self.ct_decoder2(img_s) # b, 512, 1, 1, 1
        temp.append(f3)
        feat = feat.view(feat.size(0), -1, 512) # B, 1, 512
        f3 = f3.view(feat.size(0), -1, 512) # B, 1, 512
        feat = torch.cat([feat, f3], dim=1) # B, 3, 512
        feat = feat.view(feat.size(0), feat.size(1), 8, 8, 8)
        # print(feat.shape)
        feat = self.ca(feat)
        temp.append(feat)
        feat = self.bn2(feat)
        feat = self.fc(feat.view(feat.size(0), -1))
        # if torch.isnan(feat).any():
        #     for i, j in enumerate(temp):
        #         print(i, j)
        #     return

        # f2 = f2.unsqueeze(1)
        # f3 = self.ct_decoder2(img)
        # print(f1.shape, f2.shape, feat.shape, f3.shape)
        return feat

class CCNet(nn.Module):
    def __init__(self, num_classes, model_depth=34, img_size1=(32, 32, 32), img_size2=(64, 64, 32)):
        super(CCNet, self).__init__()
        assert model_depth in [10, 18, 34, 50]
        if model_depth in [10, 18, 34]:
            hidden_dim = 512
            ca_in_channels = 2
        elif model_depth == 50:
            hidden_dim = 2048
            ca_in_channels = 8
        resnet = generate_model(model_depth)
        # img_size1 = (32, 32, 32)
        # img_size2 = (64, 64, 32)
        # self.vit1 = ViT(
        #     in_channels=1,
        #     img_size=img_size1,
        #     patch_size=16
        # )
        self.vit2 = ViT(in_channels=1, img_size=img_size2, patch_size=16, dropout_rate=0.5)

        self.ct_encoder1 = nn.Sequential(resnet.conv1,
                                        resnet.bn1,
                                        resnet.layer1,
                                        resnet.layer2,
                                        resnet.layer3,
                                        resnet.layer4,
                                        resnet.avgpool)

        # resnet = generate_model(model_depth)
        # self.ct_decoder2 = nn.Sequential(resnet.conv1,
        #                     resnet.bn1,
        #                     resnet.layer1,
        #                     resnet.layer2,
        #                     resnet.layer3,
        #                     resnet.layer4,
        #                     resnet.avgpool)
        self.proj_ct = nn.Conv3d(in_channels=resnet.fc.in_features,
                                 out_channels=768,
                                 kernel_size=1, stride=1, padding=0)
        self.att1 = Attention(768, 768)
        self.att2 = Attention(768, 768)
        self.att3 = Attention(768, 768)
        ca_in_channels = 4

        self.fuse_modal = MultiModalMultiHeadAttention(feat_dim=hidden_dim, hidden_dim=hidden_dim)
        self.ca = ChannelAttention(in_channels=ca_in_channels)

        self.fc = nn.Sequential(nn.Linear(ca_in_channels*768, 128),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128, num_classes))

        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim*2)
        self.bn2 = nn.BatchNorm3d(num_features=ca_in_channels)
    def forward(self, img, img_s):
        x2, hidden_states_out2 = self.vit2(img)
        x1 = self.ct_encoder1(img_s)
        x1 = self.proj_ct(x1) # b, 768, 1, 1, 1
        x1 = x1.view(x1.size(0), -1, 768)

        a12 = self.att1(x1, hidden_states_out2[-1])
        a8 = self.att2(x1, hidden_states_out2[-5])
        a4 = self.att2(x1, hidden_states_out2[-9]) # b, 1, 768
        feat = torch.cat((a12, a8, a4, x1), dim=1)
        # print(feat.shape)
        feat = self.ca(feat)
        # print(x1[0], hidden_states_out1[0][0])
        # print(x1.shape, x2.shape)
        # f1 = self.ct_encoder1(img) # 34: b, 512, 1, 1, 1  50: b, 2048, 1, 1, 1
        # # print(f1.shape)
        # f3 = self.ct_decoder2(img_s) # b, 512, 1, 1, 1
        # feat = self.fuse_modal(f1.view(f1.size(0), -1), f3.view(f3.size(0), -1))  # B, 1024, 1  50: b, 4096, 1
        # # print(feat.shape)
        # feat = self.bn1(feat)  # 34: B, 1024, 1 ; 50: b, 4096, 1
        # feat = feat.view(feat.size(0), -1, 512) # B, 2, 512
        # feat = feat.view(feat.size(0), feat.size(1), 8, 8, 8)
        # # print(feat.shape)
        # feat = self.ca(feat)
        # feat = self.bn2(feat)
        feat = self.fc(feat.view(feat.size(0), -1))
        # print(feat.shape)
        return feat
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 应用池化，注意去掉多余的维度
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        # 计算注意力权重
        avg_attention = self.fc(avg_out)
        max_attention = self.fc(max_out)
        channel_attention = avg_attention + max_attention

        # 调整形状以匹配输入的五维
        channel_attention = channel_attention.unsqueeze(-1)
        # print(channel_attention)
        return x * channel_attention
class myresnet101(nn.Module):
    def __init__(self, num_classes=4):
        super(myresnet101, self).__init__()
        self.num_classes = num_classes
        self.model = resnet101(weights=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        x = self.model(x)
        return x

class myresnet18(nn.Module):
    def __init__(self, num_classes=4):
        super(myresnet18, self).__init__()
        self.num_classes = num_classes
        self.model = resnet18(weights=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        x = self.model(x)
        return x
class myresnet34(nn.Module):
    def __init__(self, num_classes=4):
        super(myresnet34, self).__init__()
        self.num_classes = num_classes
        self.model = resnet34(weights=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        x = self.model(x)
        return x

class myresnet50(nn.Module):
    def __init__(self, num_classes=4):
        super(myresnet50, self).__init__()
        self.num_classes = num_classes
        self.model = resnet50(weights=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        x = self.model(x)
        return x

def get_resnet2d(rd=50, num_classes=4):
    assert rd in [18, 34, 50, 101]
    local_vars = {}
    exec(f"model = myresnet{rd}(num_classes={num_classes})", globals(), local_vars)
    return local_vars['model']

from transformers import BertModel, BertTokenizer
class TextEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, embedding_dim)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**inputs)
        pooled_output = outputs[1]
        return self.fc(pooled_output)


if __name__ == '__main__':
    # model = LungClaNet()
    # print(model)
    # img = torch.randn(2, 1, 64, 64, 32)
    # img_s = torch.randn(2, 1, 32, 32, 32)
    # # keys = torch.arange(0, 47, 1, dtype=torch.long)
    # # values = torch.randn(47)
    # # clicinal = torch.cat([torch.stack((keys, values)).unsqueeze(0), torch.stack((keys, values)).unsqueeze(0)], dim=0)
    # # print(clicinal.shape)
    # net = CCNet(3, 34)
    # out = net(img, img_s)

    # print(out.shape, out)
    model = TextEncoder(embedding_dim=256)
    text = '右肺上叶楔形及组织结节）（参见21-00722）肺中分化腺癌，呈腺泡型（90%）及乳头型（10%）。肿瘤最大径1.7cm，间质伴有纤维化，部分腺泡腔内可见少许粘液分泌物，未累及脏层胸膜，未见明确脉管瘤栓及神经侵犯，未见气腔播散（STAS）。叶、段支气管及支气管切缘未见癌。周围肺未见显著病变。'
    out = model(text)
    print(out, out.shape)





