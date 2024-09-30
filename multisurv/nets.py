import torch
import torch.nn as nn
from src.resnet import *
from bisect import bisect_left

class Attention(torch.nn.Module):
    """Attention mechanism for multimodal representation fusion."""
    def __init__(self, size):
        """
        Parameters
        ----------
        size: int
            Attention vector size, corresponding to the feature representation
            vector size.
        """
        super(Attention, self).__init__()
        self.fc = torch.nn.Linear(size, size, bias=False)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(0)  # Across feature vector stack

    def _scale_for_missing_modalities(self, x, out):
        """Scale fused feature vector up according to missing data.

        If there were all-zero data modalities (missing/dropped data for
        patient), scale feature vector values up accordingly.
        """
        batch_dim = x.shape[1]
        for i in range(batch_dim):
            patient = x[:, i, :]
            zero_dims = 0
            for modality in patient:
                if modality.sum().data == 0:
                    zero_dims += 1

            if zero_dims > 0:
                scaler = zero_dims + 1
                out[i, :] = scaler * out[i, :]

        return out

    def forward(self, x):
        scores = self.tanh(self.fc(x))
        weights = self.softmax(scores)
        out = torch.sum(x * weights, dim=0)

        out = self._scale_for_missing_modalities(x, out)

        return out
class EmbraceNet(torch.nn.Module):
    """Embracement modality feature aggregation layer."""
    def __init__(self, device='cuda:0'):
        """Embracement modality feature aggregation layer.

        Note: EmbraceNet needs to deal with mini batch elements differently
        (check missing data and adjust sampling probailities accordingly). This
        way, we take the unusual measure of considering the batch dimension in
        every operation.

        Parameters
        ----------
        device: "torch.device" object
            Device to which input data is allocated (sampling index tensor is
            allocated to the same device).
        """
        super(EmbraceNet, self).__init__()
        self.device = device

    def _get_selection_probabilities(self, d, b):
        p = torch.ones(d.size(0), b)  # Size modalities x batch

        # Handle missing data
        for i, modality in enumerate(d):
            for j, batch_element in enumerate(modality):
                if len(torch.nonzero(batch_element)) < 1:
                    p[i, j] = 0

        # Equal chances to all available modalities in each mini batch element
        m_vector = torch.sum(p, dim=0)
        p /= m_vector

        return p

    def _get_sampling_indices(self, p, c, m):
        r = torch.multinomial(
            input=p.transpose(0, 1), num_samples=c, replacement=True)
        r = torch.nn.functional.one_hot(r.long(), num_classes=m)
        r = r.permute(2, 0, 1)

        return r


    def forward(self, x):
        m, b, c = x.size()

        p = self._get_selection_probabilities(x, b)
        r = self._get_sampling_indices(p, c, m).float().to(self.device)

        d_prime = r * x
        e = d_prime.sum(dim=0)

        return e
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
                               stride=(2, 2, 2),
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
        self.n_features = self.fc.in_features


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
                # downsample = nn.Sequential(
                #     conv1x1x1(self.in_planes, planes * block.expansion, stride),
                #     nn.BatchNorm3d(planes * block.expansion))
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

class FC(nn.Module):
    "Fully-connected model to generate final output."
    def __init__(self, in_features, out_features, n_layers, dropout=True,
                 batchnorm=False, scaling_factor=4):
        super(FC, self).__init__()
        if n_layers == 1:
            layers = self._make_layer(in_features, out_features, dropout,
                                      batchnorm)
        elif n_layers > 1:
            n_neurons = self._pick_n_neurons(in_features)
            if n_neurons < out_features:
                n_neurons = out_features

            if n_layers == 2:
                layers = self._make_layer(
                    in_features, n_neurons, dropout, batchnorm=True)
                layers += self._make_layer(
                    n_neurons, out_features, dropout, batchnorm)
            else:
                for layer in range(n_layers):
                    last_layer_i = range(n_layers)[-1]

                    if layer == 0:
                        n_neurons *= scaling_factor
                        layers = self._make_layer(
                            in_features, n_neurons, dropout, batchnorm=True)
                    elif layer < last_layer_i:
                        n_in = n_neurons
                        n_neurons = self._pick_n_neurons(n_in)
                        if n_neurons < out_features:
                            n_neurons = out_features
                        layers += self._make_layer(
                            n_in, n_neurons, dropout, batchnorm=True)
                    else:
                        layers += self._make_layer(
                            n_neurons, out_features, dropout, batchnorm)
        else:
            raise ValueError('"n_layers" must be positive.')

        self.fc = nn.Sequential(*layers)

    def _make_layer(self, in_features, out_features, dropout, batchnorm):
        layer = nn.ModuleList()
        if dropout:
            layer.append(nn.Dropout())
        layer.append(nn.Linear(in_features, out_features))
        layer.append(nn.ReLU(inplace=True))
        if batchnorm:
            layer.append(nn.BatchNorm1d(out_features))

        return layer

    def _pick_n_neurons(self, n_features):
        # Pick number of features from list immediately below n input
        n_neurons = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        idx = bisect_left(n_neurons, n_features)

        return n_neurons[0 if idx == 0 else idx - 1]

    def forward(self, x):
        return self.fc(x)

class ClinicalNet(nn.Module):
    """Clinical data extractor.

    Handle continuous features and categorical feature embeddings.
    """
    def __init__(self, output_vector_size, embedding_dims=27):
        super(ClinicalNet, self).__init__()
        # Embedding layer
        # self.embedding_layers = nn.ModuleList([nn.Embedding(x, y)
        #                                        for x, y in embedding_dims])

        # n_embeddings = sum([y for x, y in embedding_dims])
        # n_continuous = 1

        # Linear Layers
        self.linear = nn.Linear(embedding_dims, 256)

        # Embedding dropout Layer
        self.embedding_dropout = nn.Dropout()

        # Continuous feature batch norm layer
        self.bn_layer = nn.BatchNorm1d(1)

        # Output Layer
        self.output_layer = FC(256, output_vector_size, 1)

    def forward(self, x):
        # categorical_x, continuous_x = x
        # categorical_x = categorical_x.to(torch.int64)

        # x = [emb_layer(categorical_x[:, i])
        #      for i, emb_layer in enumerate(self.embedding_layers)]
        # x = torch.cat(x, 1)
        # x = self.embedding_dropout(x)
        # x = self.linear(x)
        # continuous_x = self.bn_layer(continuous_x)
        # x = self.bn_layer(x)
        # x = torch.cat([x, continuous_x], 1)
        out = self.output_layer(self.linear(x))

        return out

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

class CTNet(nn.Module):
    "WSI patch feature extractor and aggregator."
    def __init__(self, output_vector_size):
        super(CTNet, self).__init__()
        self.feature_extractor = generate_model(model_depth=50)
        self.num_image_features = self.feature_extractor.n_features
        # Multiview WSI patch aggregation
        self.fc = FC(self.num_image_features, output_vector_size , 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        # view_pool = []

        # # Extract features from each patch
        # for v in x:
        #     v = self.feature_extractor(v)
        #     v = v.view(v.size(0), self.num_image_features)
        #
        #     view_pool.append(v)
        #
        # # Aggregate features from all patches
        # patch_features = torch.stack(view_pool).max(dim=1)[0]

        out = self.fc(x)

        return out
class Fusion(nn.Module):
    "Multimodal data aggregator."
    def __init__(self, method, feature_size, device):
        super(Fusion, self).__init__()
        self.method = method
        methods = ['cat', 'max', 'sum', 'prod', 'embrace', 'attention']

        if self.method not in methods:
            raise ValueError('"method" must be one of ', methods)

        if self.method == 'embrace':
            if device is None:
                raise ValueError(
                    '"device" is required if "method" is "embrace"')

            self.embrace = EmbraceNet(device=device)

        if self.method == 'attention':
            if not feature_size:
                raise ValueError(
                    '"feature_size" is required if "method" is "attention"')
            self.attention = Attention(size=feature_size)

    def forward(self, x):
        if self.method == 'attention':
            out = self.attention(x)
        if self.method == 'cat':
            out = torch.cat([m for m in x], dim=1)
        if self.method == 'max':
            out = x.max(dim=0)[0]
        if self.method == 'sum':
            out = x.sum(dim=0)
        if self.method == 'prod':
            out = x.prod(dim=0)
        if self.method == 'embrace':
            out = self.embrace(x)

        return out

class MultiSurv(torch.nn.Module):
    """Deep Learning model for MULTImodal pan-cancer SURVival prediction."""
    def __init__(self, fusion_method='max', clinical_length=27,
                 n_output_intervals=None, device=None):
        super(MultiSurv, self).__init__()
        # self.data_modalities = data_modalities
        self.mfs = modality_feature_size = 512
        # valid_mods = ['clinical', 'wsi', 'mRNA', 'miRNA', 'DNAm', 'CNV']
        # assert all(mod in valid_mods for mod in data_modalities), \
        #         f'Accepted input data modalitites are: {valid_mods}'

        # assert len(data_modalities) > 0, 'At least one input must be provided.'

        if fusion_method == 'cat':
            self.num_features = 0
        else:
            self.num_features = self.mfs

        self.submodels = {}

        # Clinical -----------------------------------------------------------#
        self.clinical_submodel = ClinicalNet(
            output_vector_size=self.mfs, embedding_dims=clinical_length)
        self.submodels['clinical'] = self.clinical_submodel

        if fusion_method == 'cat':
            self.num_features += self.mfs

        self.ct_submodel = CTNet(output_vector_size=self.mfs)
        self.submodels['ct'] = self.ct_submodel

        if fusion_method == 'cat':
            self.num_features += self.mfs

        # Instantiate multimodal aggregator ----------------------------------#
        # if len(data_modalities) > 1:
        self.aggregator = Fusion(fusion_method, self.mfs, device)

        # Fully-connected and risk layers ------------------------------------#
        n_fc_layers = 4
        n_neurons = 512

        self.fc_block = FC(
            in_features=self.num_features, out_features=n_neurons,
            n_layers=n_fc_layers)

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_neurons,
                            out_features=3)
            # torch.nn.Sigmoid()
        )

    def forward(self, data):
        multimodal_features = []
        multimodal_features.append(self.ct_submodel(data['image']))
        multimodal_features.append(self.clinical_submodel(data['clinical']))
        # Run data through modality sub-models (generate feature vectors) ----#
        # for modality in x:
        #     multimodal_features += (self.submodels[modality](x[modality]),)

        # Feature fusion/aggregation -----------------------------------------#
        if len(multimodal_features) > 1:
            x = self.aggregator(torch.stack(multimodal_features))
            feature_repr = {'modalities': multimodal_features, 'fused': x}
        else:  # skip if running unimodal data
            x = multimodal_features[0]
            feature_repr = {'modalities': multimodal_features[0]}

        # Outputs ------------------------------------------------------------#
        # print(x.shape)
        x = self.fc_block(x)
        risk = self.risk_layer(x)

        # Return non-zero features (not missing input data)
        # output_features = tuple()
        #
        # for modality in multimodal_features:
        #     modality_features = torch.stack(
        #         [batch_element for batch_element in modality
        #          if batch_element.sum() != 0])
        #     output_features += modality_features,
        #
        # feature_repr['modalities'] = output_features

        return risk
if __name__ == '__main__':
    ct = torch.randn(2, 1, 128, 128, 32)
    cli = torch.randn(2, 27)
    # net = CTNet(output_vector_size=512)
    net = MultiSurv()
    data = {'image': ct, 'clinical': cli}
    print(net(data).shape)