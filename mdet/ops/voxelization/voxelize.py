# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Function
from .voxelization import OpVoxelization


class _Voxelize(Function):

    @staticmethod
    def forward(ctx,
                points,
                point_range,
                voxel_size,
                max_points=32,
                max_voxels=20000,
                reduce_type=0):
        """convert kitti points NxD(D>=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other feature
            point_range: [6] list/tuple or array, float. indicate valid point coordinates
                format: xyzxyz, minmax
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.

        Returns:
            voxels: [max_voxels, max_points, ndim] float tensor.
            coords: [max_voxels, 3] int32 tensor, indicate the voxel coordinate
            point_num: [max_voxels] int32 tensor, indicate the valid point num in each voxel
            voxel_num: [] int32 tensor, indicate the valid voxel num
        """
        assert max_points >= 1
        voxels = points.new_zeros(
            size=(max_voxels, max_points, points.size(1)))
        coords = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        point_num = points.new_zeros(size=(max_voxels, ), dtype=torch.int)
        voxel_num = points.new_zeros(size=(), dtype=torch.int)

        OpVoxelization(points, point_range, voxel_size, max_points, max_voxels, reduce_type,
                   voxels, coords, point_num, voxel_num)

        return voxels, coords, point_num, voxel_num

    @staticmethod
    def symbolic(g, points, voxel_size, coors_range, max_points, max_voxels, reduce_type):
        return g.op(
            'custom_ops::Voxelization',
            points,
            voxel_size_f=voxel_size,
            point_range_f=coors_range,
            max_points_i=max_points,
            max_voxels_i=max_voxels,
            reduce_type_i=reduce_type,
            outputs=4)


Voxelize = _Voxelize.apply
