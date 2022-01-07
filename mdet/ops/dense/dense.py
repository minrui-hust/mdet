import torch
from torch import nn
from torch.autograd import Function


class Dense(Function):

    @staticmethod
    def forward(ctx, pillars, coords, valid_voxels, voxel_reso):
        nx, ny = voxel_reso[:2]

        canvas = pillars.new_zeros(pillars.size(1), nx*ny)

        pillars = pillars[:valid_voxels]
        coords = coords[:valid_voxels]

        indices = coords[:, 1] * nx + coords[:, 2]
        indices = indices.long()
        pillars = pillars.t()
        canvas[:, indices] = pillars
        canvas = canvas.view(-1, ny, nx)
        return canvas

    @staticmethod
    def symbolic(g, pillars, cords, valid_pillars, pillar_reso):
        return g.op(
            'custom_ops::Dense',
            pillars,
            cords,
            valid_pillars,
            pillar_reso_i=pillar_reso,
            outputs=1)

dense = Dense.apply
