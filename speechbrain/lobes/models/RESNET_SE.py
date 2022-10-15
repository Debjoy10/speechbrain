import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .ResNetBlocks import *

class ResNetSE(nn.Module):
    """An implementation of RESNET-SE for speaker embedding.
    Implementation adapted from https://github.com/TaoRuijie/Speaker-Recognition-Demo/blob/main/models/ResNetSE34V2.py

    Arguments
    ---------
    block : torch.nn.Module
        Basic Block of ResNet
    layers : list of ints
        Number of layers in each block
    num_filters : list of ints
        Number of filters in each block
    nOut : int
        Output dimension
    n_mels : int
        Input Dimension

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ResNetSE(n_mels=80, nOut=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(self, block = SEBasicBlock, layers = [3, 4, 6, 3], num_filters = [32, 64, 128, 256], nOut=192, encoder_type='ASP', n_mels=64, **kwargs):
        super(ResNetSE, self).__init__()
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        lyrs = []
        s = (1, 1)
        for li in range(len(num_filters)):
            lyrs.append(self._make_layer(block, num_filters[li],\
                                           layers[li], stride=s))
            s = (2, 2)
        self.layers = nn.ModuleList(lyrs)

        outmap_size = int(self.n_mels/8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[-1] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[-1] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
            )

        if self.encoder_type == "SAP":
            out_dim = num_filters[-1] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[-1] * outmap_size * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):

        x = x.unsqueeze(1).transpose(2, 3)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        for layer in self.layers:
            x = layer(x)
        
        x = x.reshape(x.size()[0],-1,x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,sg),1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x).unsqueeze(1)

        return x