import torch
import torch.nn as nn
import torch.nn.functional as F

from mdet.model.utils import build_norm
from mdet.utils.factory import FI
import spconv.pytorch as spconv
from spconv.pytorch import SparseConv3d, SubMConv3d


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=dict(type="BatchNorm1d", eps=1e-3, momentum=0.01),
        downsample=None,
        indice_key=None,
    ):
        super().__init__()

        self.conv1 = spconv.SubMConv3d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            indice_key=indice_key,
        )
        self.bn1 = build_norm(norm_cfg, planes)

        self.conv2 = spconv.SubMConv3d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            indice_key=indice_key,
        )
        self.bn2 = build_norm(norm_cfg, planes)

        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(F.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(F.relu(out.features))

        return out


@FI.register
class SparseResNetFHD(nn.Module):
    def __init__(self,
                 in_channels=4,
                 norm_cfg=dict(type="BatchNorm1d", eps=1e-3, momentum=0.01),
                 ):
        super().__init__()

        self.conv_input = spconv.SparseSequential(
            SubMConv3d(in_channels, 16, 3, bias=False, indice_key="res0"),
            build_norm(norm_cfg, 16),
            nn.ReLU(inplace=True),
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False),
            build_norm(norm_cfg, 32),
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False),
            build_norm(norm_cfg, 64),
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(64, 128, 3, stride=2, padding=[0, 1, 1], bias=False),
            build_norm(norm_cfg, 128),
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1),
                         bias=False),
            build_norm(norm_cfg, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxelization_result):
        r'''
        voxelization_result: a dict at least contain:
            {
                voxels: shape NxF
                coords: shape Nx4, (sample_id, z, y, x), int32
                shape: (z_reso, y_reso, x_reso)
                batch_size: scalar
            }
        '''
        voxels = voxelization_result['voxels']
        coords = voxelization_result['coords']
        shape = voxelization_result['shape']
        batch_size = voxelization_result['batch_size']

        if coords.shape[1] == 3:
            # padding for sample id, (z,y,x) - >(sample_id, z, y, x)
            coords = torch.cat(
                [coords.new_zeros((coords.shape[0], 1)), coords], dim=1)

        # extend 1 on z dimension
        shape = [shape[0] + 1, shape[1], shape[2]]

        sp_in = spconv.SparseConvTensor(voxels, coords, shape, batch_size)

        x = self.conv_input(sp_in)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        sp_out = self.extra_conv(x_conv4)

        out = sp_out.dense()

        return out.view(*out.shape[:1], -1, *out.shape[3:])


@FI.register
class SparseResNetUHD(nn.Module):
    def __init__(self,
                 in_channels=4,
                 norm_cfg=dict(type="BatchNorm1d", eps=1e-3, momentum=0.01),
                 ):
        super().__init__()

        self.conv_input = spconv.SparseSequential(
            SubMConv3d(in_channels, 16, 3, bias=False, indice_key="res0"),
            build_norm(norm_cfg, 16),
            nn.ReLU(inplace=True),
        )

        self.conv1 = spconv.SparseSequential(
            SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False),
            build_norm(norm_cfg, 32),
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(32, 32, 3, stride=1, padding=1, bias=False),
            build_norm(norm_cfg, 32),
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False),
            build_norm(norm_cfg, 64),
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(64, 64, 3, stride=1, padding=[0, 1, 1], bias=False),
            build_norm(norm_cfg, 64),
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res4"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res4"),
        )

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), bias=False),
            build_norm(norm_cfg, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxelization_result):
        r'''
        voxelization_result: a dict at least contain:
            {
                voxels: shape NxF
                coords: shape Nx4, (sample_id, z, y, x), int32
                shape: (z_reso, y_reso, x_reso)
                batch_size: scalar
            }
        '''
        voxels = voxelization_result['voxels']
        coords = voxelization_result['coords']
        shape = voxelization_result['shape']
        batch_size = voxelization_result['batch_size']

        if coords.shape[1] == 3:
            # padding for sample id, (z,y,x) - >(sample_id, z, y, x)
            coords = torch.cat(
                [coords.new_zeros((coords.shape[0], 1)), coords], dim=1)

        # extend 1 on z dimension
        shape = [shape[0] + 1, shape[1], shape[2]]

        sp_in = spconv.SparseConvTensor(voxels, coords, shape, batch_size)

        x = self.conv_input(sp_in)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        sp_out = self.extra_conv(x_conv4)

        out = sp_out.dense()

        return out.view(*out.shape[:1], -1, *out.shape[3:])
