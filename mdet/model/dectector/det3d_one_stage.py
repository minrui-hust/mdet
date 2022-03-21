import torch
import torch.nn as nn
import torch.nn.functional as ff

from mai.utils import FI
from mai.model import BaseModule


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
        bb3d_out = self.backbone3d(voxel_out)
        bb2d_out = self.backbone2d(bb3d_out)
        neck_out = self.neck(bb2d_out)
        head_out = self.head(neck_out)
        return head_out

    def forward_infer(self, points):
        voxel_out = self.voxelization(points)
        bb3d_out = self.backbone3d(voxel_out)
        bb2d_out = self.backbone2d(bb3d_out)
        neck_out = self.neck(bb2d_out)
        head_out = self.head(neck_out)
        return head_out
