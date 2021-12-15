import torch
import torch.nn as nn
import torch.nn.functional as F

from mdet.model import BaseModule
from mdet.utils.factory import FI


@FI.register
class SECOND(BaseModule):
    """Backbone network for PointPillars.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(layer_strides) == len(layer_nums) == len(out_channels)

        in_channels = [in_channels] + out_channels[:-1]

        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Conv2d(in_channels[i], out_channels[i], 3, stride=layer_strides[i],padding=1),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_num):
                block.append(nn.Conv2d(out_channels[i], out_channels[i], 3, padding=1))
                block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                block.append(nn.ReLU(inplace=True))
            block = nn.Sequential(*block)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs
