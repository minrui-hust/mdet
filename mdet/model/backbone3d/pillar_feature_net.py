import torch
import torch.nn as nn
import torch.nn.functional as F

from mdet.model import BaseModule
from mdet.utils.factory import FI
from mdet.model.utils import construct_mask

import math


@FI.register
class PillarFeatureNet(BaseModule):
    r'''Pillar Feature Net. convert the voxelization result to 2d persudo image
    '''

    def __init__(self,
                 pillar_feat=[('position',3), ('attribute', 1)],
                 voxel_size=(0.2, 0.2, 4),
                 point_range=(0, -40, -3, 70.4, 40, 1),
                 pfn_channels=(64, )):
        super(PillarFeatureNet, self).__init__()
        assert len(pfn_channels) > 0
        
        self.pillar_feat = pillar_feat
        in_channels = sum([f[1] for f in self.pillar_feat])
        assert in_channels > 0

        pfn_channels = [in_channels] + list(pfn_channels)
        self.pfn_layers = nn.ModuleList([PFNLayer(pfn_channels[i], pfn_channels[i+1], last_layer= (i == len(pfn_channels)-2))
                                         for i in range(len(pfn_channels)-1)])

        self.voxel_size = voxel_size
        self.voxel_offset = (-point_range[0] + voxel_size[0]/2, 
                             -point_range[1] + voxel_size[1]/2)
        self.voxel_reso = (int(math.floor((point_range[3]-point_range[0])/voxel_size[0])), 
                           int(math.floor((point_range[4]-point_range[1])/voxel_size[1])))

    def forward(self, voxelization_result):
        """Forward function.

        Args:
            voxelization_result (list of tuple): result from voxelization

        Returns:
            torch.Tensor: Features of dense 2d feature image
        """

        # collate voxelization into tensors
        voxels, coords, point_nums, voxel_nums = self.colloate(voxelization_result)

        # construct pillar feature from raw points
        pillar_feature = self.construct_pillar(voxels, coords, point_nums)

        # pass through pfn layers
        mask = construct_mask(point_nums, voxels.size(1), inverse=True)
        for pfn in self.pfn_layers:
            pillar_feature = pfn(pillar_feature, mask)

        # in shape [batch, W, H, channels]
        feature_image = self.scatter(pillar_feature, coords, voxel_nums)

        return feature_image

    def colloate(self, voxelization_result):
        r'''
        collate the voxelization output to tensors
        '''

        voxels, coords, point_nums, voxel_nums = [],[],[],[]
        for voxel, coord, point_num, voxel_num in voxelization_result:
            voxels.append(voxel)
            coords.append(coord)
            point_nums.append(point_num)
            voxel_nums.append(voxel_num)

        # stack along batch dimension
        voxels = torch.stack(voxels, dim=0)
        coords = torch.stack(coords, dim=0)
        point_nums = torch.stack(point_nums, dim=0)
        voxel_nums = torch.stack(voxel_nums, dim=0)

        return voxels, coords, point_nums, voxel_nums

    def construct_pillar(self, voxels, coords, point_nums):
        r'''
        Construct pillar from raw points in a voxel, invalid points has value of zero
        voxels: [batch_size, max_pillar_num, max_point_num, point_dimension]
        coords: [batch_size, max_pillar_num, 3], in z,y,x order
        point_nums: [batch_size]
        '''

        position = voxels[...,:3]
        attribute = voxels[...,3:]
        feature_list = []
        for feat_name, _ in self.pillar_feat:
            if feat_name == 'attribute':
                feature_list.append(attribute) # original point attribute
            elif feat_name == 'position': # original point position
                feature_list.append(position)
            elif feat_name == 'mean_offset': # offset to pillar mean
               pillar_mean = position.sum(dim=1, keepdim=True) / point_nums.type_as(voxels).view(-1, 1, 1)
               mean_offset = position - pillar_mean
               feature_list.append(mean_offset)
            elif feat_name == 'center_offset': # offset to voxel center
                voxel_size = torch.tensor([self.voxel_size[0], self.voxel_size[1]], 
                        dtype=voxels.dtype, device=voxels.device)
                voxel_offset = torch.tensor([self.voxel_offset[0], self.voxel_offset[1]], 
                        dtype=voxels.dtype, device=voxels.device)
                pillar_center = coords[..., [2, 1]].type_as(voxels) * voxel_size + voxel_offset
                center_offset = position[..., :2] - pillar_center
                feature_list.append(center_offset)
            elif feat_name == 'distance': # distance to origin
                distance = torch.norm(position, 2, 2, keepdim=True)
                feature_list.append(distance)
            else:
                raise NotImplementedError
        return torch.cat(feature_list, dim=-1)

    def scatter(self, pillar_feature, coords, voxel_nums):
        r'''
        pillar_feature: shape [B, max_pillar_num, feature_size] 
        coords: shape [B, max_pillar_num, 4]
        voxel_nums: shape [B]
        '''
        batch_size, feature_size = voxel_nums.size(0), pillar_feature.size(2)
        w, h = self.voxel_reso

        canvas = torch.zeros(batch_size, feature_size, w*h,
                dtype=pillar_feature.dtype, device=pillar_feature.device)
        for i in range(batch_size):
            coord = coords[i, :voxel_nums[i]]
            index = coord[:, 1] * w + coord[:, 2]
            feat = pillar_feature[i, :voxel_nums[i]]
            canvas[i, :, index] = feat.t()

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

        self.linear = nn.Linear(self.in_channels, self.hidden_channels, bias=False)
        self.norm = nn.LayerNorm(self.hidden_channels)

    def forward(self, x, mask=None):

        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)

        if mask:
            x.masked_fill_(mask.unsqueeze(-1), float('-inf'))

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max.squeeze(1)
        else:
            return torch.cat([x, x_max.expand_as(x)], dim=-1)



