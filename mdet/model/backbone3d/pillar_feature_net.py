import math

import torch
from torch.autograd.profiler import record_function
import torch.nn as nn
import torch.nn.functional as F

from mdet.model import BaseModule
from mdet.model.utils import construct_mask
from mdet.ops.dense import dense
from mdet.utils.factory import FI
from mdet.utils.misc import is_nan_or_inf


@FI.register
class PillarFeatureNet(BaseModule):
    r'''Pillar Feature Net. convert the voxelization result to 2d persudo image
    '''

    def __init__(self,
                 pillar_feat=[('position', 3), ('attribute', 1)],
                 point_range=(0, -40, -3, 70.4, 40, 1),
                 voxel_size=(0.2, 0.2, 4),
                 voxel_reso=(100, 100, 1),
                 pfn_channels=(64, )):
        super(PillarFeatureNet, self).__init__()
        assert len(pfn_channels) > 0

        self.pillar_feat = pillar_feat
        in_channels = sum([f[1] for f in self.pillar_feat])
        assert in_channels > 0

        pfn_channels = [in_channels] + list(pfn_channels)
        self.pfn_layers = nn.ModuleList([PFNLayerBN(pfn_channels[i], pfn_channels[i+1], last_layer=(i == len(pfn_channels)-2))
                                         for i in range(len(pfn_channels)-1)])

        self.voxel_reso = voxel_reso[:2]
        self.voxel_size = voxel_size[:2]
        self.voxel_offset = (point_range[0] + voxel_size[0]/2,
                             point_range[1] + voxel_size[1]/2)

    def forward_train(self, voxelization_result):
        """Forward function.

        Args:
            voxelization_result: result from voxelization, which is a dict contain
                {
                 'voxels': NxPxF
                 'coords': Nx4, (sample_id, z, y, x)
                 'point_nums': N
                 'shape': (reso_z, reso_y, reso_x)
                 'batch_size: number of samples'
                }

        Returns:
            torch.Tensor: Features of dense 2d feature image
        """
        voxels = voxelization_result['voxels']
        coords = voxelization_result['coords']
        point_nums = voxelization_result['point_nums']
        batch_size = voxelization_result['batch_size']

        # construct pillar feature from raw points
        with record_function("construct_pillar"):
            pillar_feature = self.construct_pillar(voxels, coords, point_nums)

        # pass through pfn layers
        with record_function("pillar_pfn"):
            mask = construct_mask(point_nums, voxels.size(-2), inverse=True)
            for pfn in self.pfn_layers:
                pillar_feature = pfn(pillar_feature, mask)

        # in shape [batch, W, H, channels]
        with record_function("pillar_scatter"):
            feature_image = self.scatter(pillar_feature, coords, batch_size)

        return feature_image

    def forward_infer(self, voxelization_result):
        voxels = voxelization_result['voxels']
        coords = voxelization_result['coords']
        point_nums = voxelization_result['point_nums']
        voxel_nums = voxelization_result['voxel_nums']

        # construct pillar feature from raw points
        pillar_feature = self.construct_pillar(voxels, coords, point_nums)

        # pass through pfn layers
        mask = construct_mask(point_nums, voxels.size(-2), inverse=True)
        for pfn in self.pfn_layers:
            pillar_feature = pfn(pillar_feature, mask)

        # in shape [1, channels, W, H]
        feature_image = dense(pillar_feature, coords,
                              voxel_nums, self.voxel_reso).unsqueeze(0)

        return feature_image

    def construct_pillar(self, voxels, coords, point_nums):
        r'''
        Construct pillar from raw points in a voxel, invalid points has value of zero
        voxels: [max_pillar_num, max_point_num, point_dimension]
        coords: [max_pillar_num, 4], in [sample_id],z,y,x order
        point_nums: [max_pillar_num]
        '''

        if coords.shape[1] == 4:
            coords = coords[:, [3, 2]]
        else:
            coords = coords[:, [2, 1]]

        position = voxels[..., :3]
        attribute = voxels[..., 3:]
        feature_list = []
        for feat_name, _ in self.pillar_feat:
            if feat_name == 'attribute':
                feature_list.append(attribute)  # original point attribute
            elif feat_name == 'position':  # original point position
                feature_list.append(position)
            elif feat_name == 'mean_offset':  # offset to pillar mean
                pillar_mean = position.sum(
                    dim=-2, keepdim=True) / point_nums.type_as(voxels).unsqueeze(-1).unsqueeze(-1)
                mean_offset = position - pillar_mean
                feature_list.append(mean_offset)
            elif feat_name == 'center_offset':  # offset to voxel center
                voxel_size = torch.tensor([self.voxel_size[0], self.voxel_size[1]],
                                          dtype=voxels.dtype, device=voxels.device)
                voxel_offset = torch.tensor([self.voxel_offset[0], self.voxel_offset[1]],
                                            dtype=voxels.dtype, device=voxels.device)
                pillar_center = coords.type_as(
                    voxels) * voxel_size + voxel_offset
                center_offset = position[..., :2] - pillar_center.unsqueeze(-2)
                feature_list.append(center_offset)
            elif feat_name == 'distance':  # distance to origin
                distance = torch.norm(position, 2, -1, keepdim=True)
                feature_list.append(distance)
            else:
                raise NotImplementedError
        voxels = torch.cat(feature_list, dim=-1)

        # [N, point_num, feature_num] to [N, feature_num, point_num]
        return voxels.transpose(-1, -2)

    def scatter(self, pillar_feature, coords, batch_size):
        r'''
        pillar_feature: shape [B, max_pillar_num, feature_size] 
        coords: shape [B, max_pillar_num, 4]
        voxel_nums: shape [B]
        '''
        feature_size = pillar_feature.size(-1)
        w, h = self.voxel_reso

        canvas = torch.zeros(batch_size, feature_size, w*h,
                             dtype=pillar_feature.dtype, device=pillar_feature.device)

        for i in range(batch_size):
            mask = coords[:, 0] == i
            coord = coords[mask]
            index = (coord[:, 2]*w + coord[:, 3]).long()
            canvas[i, :, index] = pillar_feature[mask].t()

        return canvas.view(batch_size, feature_size, h, w)


class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, last_layer=False):
        r'''
        '''
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = out_channels
        self.last_pfn = last_layer
        if not self.last_pfn:
            self.hidden_channels = self.hidden_channels // 2

        self.linear = nn.Linear(
            self.in_channels, self.hidden_channels, bias=False)
        self.norm = nn.LayerNorm(self.hidden_channels)

    def forward(self, x, mask=None):
        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), float('-inf'))

        x_max = torch.max(x, dim=-2, keepdim=True)[0]

        if self.last_pfn:
            return x_max.squeeze(-2)
        else:
            return torch.cat([x, x_max.expand_as(x)], dim=-1)


class PFNLayerBN(nn.Module):
    def __init__(self, in_channels, out_channels, last_layer=False):
        r'''
        '''
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = out_channels
        self.last_pfn = last_layer
        if not self.last_pfn:
            self.hidden_channels = self.hidden_channels // 2

        self.linear = nn.Conv1d(
            self.in_channels, self.hidden_channels, 1, bias=False)
        self.norm = nn.BatchNorm1d(self.hidden_channels)

    def forward(self, x, mask=None):
        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)

        if mask is not None:  # mask is N x max_points, unsqueeze to N x 1 x max_points
            x = x.masked_fill(mask.unsqueeze(-2), float('-inf'))

        # x.shape N x C x max_points
        x_max = torch.max(x, dim=-1, keepdim=True)[0]

        if self.last_pfn:
            return x_max.squeeze(-1)
        else:
            return torch.cat([x, x_max.expand_as(x)], dim=-2)
