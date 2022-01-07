import torch
from torch.autograd import Function
from .voxelization import OpVoxelization
from torch._C import Graph


class _Voxelize(Function):
    @staticmethod
    def forward(ctx,
                points,
                point_range,
                voxel_size,
                voxel_reso,
                max_points=32,
                max_voxels=20000,
                reduce_type=None
                ):
        r'''convert kitti points NxD(D>=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other feature
            point_range: [6] list/tuple or array, float. indicate valid point coordinates
                format: xyzxyz, minmax
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            voxel_reso: [3] list/tuple or array, int32. xyz, indicate voxel
                resolution(how many voxels on each dimension)
            max_points: int. indicate maximum points contained in a voxel.
                0 means no limit, which is only used when reduction enabled.
            max_voxels: int. indicate maximum voxels this function create.
                Users should shuffle points before call this function because max_voxels may drop points.
            reduce_type: reduction method of points in a voxel, available type are:
                'mean', 'first', 'nearest'.

        Returns:
            voxels: [max_voxels, max_points, ndim] or [max_voxels, ndim] float32 tensor.
            coords: [max_voxels, 3] int32 tensor, indicate the voxel coordinate
            point_num: [max_voxels] int32 tensor, indicate the valid point num in each voxel.
                when reduction enabled, this means max points used in reduction
            voxel_num: [] int32 tensor, indicate the valid voxel num
        '''
        assert max_points >= 0

        voxels = points.new_empty(
            size=(max_voxels, max_points if not reduce_type else 1, points.size(-1)))
        coords = points.new_empty(size=(max_voxels, 3), dtype=torch.int)
        point_num = points.new_empty(size=(max_voxels, ), dtype=torch.int)
        voxel_num = points.new_empty(size=(), dtype=torch.int)

        OpVoxelization(points, point_range, voxel_size, voxel_reso, max_points,
                       max_voxels, reduce_type, voxels, coords, point_num, voxel_num)

        return voxels, coords, point_num, voxel_num

    @staticmethod
    def symbolic(g, points, point_range, voxel_size, voxel_reso, max_points, max_voxels, reduce_type):
        return g.op(
            'custom_ops::Voxelization',
            points,
            point_range_f=point_range,
            voxel_size_f=voxel_size,
            voxel_reso_i=voxel_reso,
            max_points_i=max_points,
            max_voxels_i=max_voxels,
            reduce_type_i=reduce_type,
            outputs=4)


def Voxelize(points,
             point_range,
             voxel_size,
             voxel_reso,
             max_points=32,
             max_voxels=20000,
             reduce_type=None,
             keep_dim=False):

    if reduce_type == 'first':  # force max_points to 1 when reduce_type is 'first'
        max_points = 1

    if reduce_type is None:
        reduce_type = 0
    elif reduce_type == 'mean':
        reduce_type = 1
    elif reduce_type == 'first':
        reduce_type = 2
    elif reduce_type == 'nearest':
        reduce_type = 3
    else:
        raise NotImplementedError

    voxels, coords, point_num, voxel_num = _Voxelize.apply(points,
                                                           point_range,
                                                           voxel_size,
                                                           voxel_reso,
                                                           max_points,
                                                           max_voxels,
                                                           reduce_type)

    if reduce_type > 0 and not keep_dim:
        voxels = voxels.squeeze(-2)

    return voxels, coords, point_num, voxel_num
