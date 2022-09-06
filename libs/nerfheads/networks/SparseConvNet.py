# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())


class SparseConvNet(nn.Module):
    def __init__(self, n_layers=4, in_dim=16, out_dim=[32, 32, 32, 32]):
        super(SparseConvNet, self).__init__()
        self.n_layers = n_layers
        assert len(out_dim) == self.n_layers
        self.net = nn.ModuleList([])
        for i in range(n_layers):
            if i == 0:
                self.net.append(double_conv(in_dim, in_dim, 'subm'+str(i)))
                self.net.append(stride_conv(in_dim, out_dim[0], 'down'+str(i)))
            else:
                self.net.append(double_conv(out_dim[i-1], out_dim[i-1], 'subm'+str(i)))
                self.net.append(stride_conv(out_dim[i-1], out_dim[i], 'down'+str(i)))
        self.net.append(double_conv(out_dim[-1], out_dim[-1], 'subm'+str(n_layers)))

    def forward(self, x, grid_coords=None):
        x = self.net[0](x)
        features = []
        for i in range(self.n_layers):
            x = self.net[2*i+1](x)
            x = self.net[2*i+2](x)
            feat = x.dense()
            if grid_coords is not None:
                features.append(F.grid_sample(feat,
                                              grid_coords,
                                              padding_mode='zeros',
                                              align_corners=True))
            else:
                features.append(feat)

        if grid_coords is not None:
            features = torch.cat(features, dim=1)
            features = features.view(features.size(0), -1, features.size(4))

        return features

    def encode(self, x, threshold=0.1):
        x = self.net[0](x)
        masks3d = []
        features = []
        for i in range(self.n_layers):
            x = self.net[2*i+1](x)
            x = self.net[2*i+2](x)
            feat = x.dense()
            features.append(feat)
        for i in range(len(features)):
            msk = features[i][0].sum(dim=0)
            masks3d.append(F.interpolate(msk[None, None, ...], features[0].shape[-3:])[0, 0])
        masks3d = torch.stack(masks3d, dim=0).sum(dim=0)
        self.masks3d = masks3d
        self.mask_xyz = torch.stack(torch.where(masks3d > threshold), dim=0).permute(
            1, 0).flip(-1).float() * 2.0
        
        self.features = features