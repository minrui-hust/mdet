import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mdet.model import BaseModule
from mdet.utils.factory import FI


@FI.register
class SECONDFPN(BaseModule):
    """FPN used in PointPillars.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4]):
        super().__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)

        deblocks = []
        for in_channel, out_channel, stride in zip(in_channels, out_channels, upsample_strides):
            if stride > 1:
                upsample_layer = nn.ConvTranspose2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                    bias=False)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                    bias=False)

            deblock = nn.Sequential(upsample_layer,
                                    nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)


    def forward(self, x):
        assert len(x) == len(self.deblocks)

        ups = [deblock(d) for d, deblock in zip(x, self.deblocks) ] 

        if len(ups) >1:
            return torch.cat(ups, dim=1)
        else:
            return ups[0]
            
