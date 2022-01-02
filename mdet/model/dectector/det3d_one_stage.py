import torch
import torch.nn as nn
import torch.nn.functional as ff

from mdet.utils.factory import FI
from mdet.model.base_module import BaseModule
from mdet.utils.misc import is_nan_or_inf


@FI.register
class Det3dOneStage(BaseModule):
    def __init__(self, voxelization, backbone3d, backbone2d, neck, head):
        super().__init__()

        self.voxelization = FI.create(voxelization)
        self.backbone3d = FI.create(backbone3d)
        self.backbone2d = FI.create(backbone2d)
        self.neck = FI.create(neck)
        self.head = FI.create(head)

    def forward_train(self, batch):
        voxel_out = self.voxelization(batch)
        #  is_nan_or_inf(voxel_out, 'voxel_out')
        bb3d_out = self.backbone3d(voxel_out)
        #  is_nan_or_inf(bb3d_out, 'bb3d_out')
        bb2d_out = self.backbone2d(bb3d_out)
        #  is_nan_or_inf(bb2d_out, 'bb2d_out')
        neck_out = self.neck(bb2d_out)
        #  is_nan_or_inf(neck_out, 'neck_out')
        head_out = self.head(neck_out)
        #  is_nan_or_inf(head_out, 'head_out')
        return head_out
