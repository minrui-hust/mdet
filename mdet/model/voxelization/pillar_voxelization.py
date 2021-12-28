import torch
from mdet.model import BaseModule
from mdet.ops.voxelization import Voxelize
from mdet.utils.factory import FI


@FI.register
class PillarVoxelization(BaseModule):
    def __init__(self, point_range, voxel_size, voxel_reso, max_points, max_voxels):
        super().__init__()

        self.point_range = point_range
        self.voxel_size = voxel_size
        self.voxel_reso = voxel_reso
        self.max_points = max_points
        self.max_voxels = max_voxels

    @torch.no_grad()
    def forward_train(self, points):
        return Voxelize(points, self.point_range, self.voxel_size, self.voxel_reso, self.max_points, self.max_voxels)
