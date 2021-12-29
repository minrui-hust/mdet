import torch
import torch.nn.functional as F
from torch.autograd.profiler import record_function

from mdet.model import BaseModule
from mdet.ops.voxelization import Voxelize
from mdet.utils.factory import FI


@FI.register
class PillarVoxelization(BaseModule):
    def __init__(self, point_range, voxel_size, voxel_reso, max_points, max_voxels, reduce_type=None, keep_dim=False):
        super().__init__()

        self.point_range = point_range
        self.voxel_size = voxel_size
        self.voxel_reso = voxel_reso
        self.max_points = max_points
        self.max_voxels = max_voxels
        self.reduce_type = reduce_type
        self.keep_dim = keep_dim

    @torch.no_grad()
    def forward_train(self, batch):
        # iterate on pcds
        voxel_out_list = [
            Voxelize(points,
                     self.point_range,
                     self.voxel_size,
                     self.voxel_reso,
                     self.max_points,
                     self.max_voxels,
                     self.reduce_type,
                     self.keep_dim)
            for points in batch['input']['pcd']
        ]

        # collate voxelization output list
        with record_function("collate_voxel"):
            voxels, coords, point_nums = self.colloate(voxel_out_list)

        return dict(
            voxels=voxels,
            coords=coords,
            point_nums=point_nums,
            shape=list(reversed(self.voxel_reso)),
            batch_size=batch['_info_']['size'],
        )

    def colloate(self, voxel_out_list):
        r'''
        collate the voxelization output to tensors
        '''
        voxels, coords, point_nums = [], [], []
        for i, (voxel, coord, point_num, voxel_num) in enumerate(voxel_out_list):
            voxels.append(voxel[:voxel_num])
            coords.append(F.pad(coord[:voxel_num], (1, 0), value=i))
            point_nums.append(point_num[:voxel_num])

        # stack along batch dimension
        voxels = torch.cat(voxels, dim=0)
        coords = torch.cat(coords, dim=0)
        point_nums = torch.cat(point_nums, dim=0)

        return voxels, coords, point_nums
