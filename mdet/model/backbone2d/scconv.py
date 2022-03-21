import torch
import torch.nn as nn
import torch.nn.functional as F

from mai.model import BaseModule
from mai.utils import FI


@FI.register
class SCConvNet(BaseModule):
    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(layer_strides) == len(layer_nums) == len(out_channels)

        in_channels = [in_channels] + out_channels[:-1]

        blocks = []
        for i in range(len(layer_nums)):
            blocks.append(SCBlock(
                in_channels[i], out_channels[i], stride=layer_strides[i], layer_num=layer_nums[i]))
        self.blocks = nn.ModuleList(blocks)

    def forward_train(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs


class SCBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, stride=1, layer_num=2):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        sc_blocks = []

        for _ in range(layer_num):
            sc_blocks.append(SCConv(out_channels))

        self.sc_blocks = nn.Sequential(*sc_blocks)

    def forward(self, x):
        out = self.input_block(x)
        out = self.sc_blocks(out)
        return out


class SCConv(nn.Module):
    def __init__(self, channels, pooling_r=4):
        super().__init__()
        assert(channels % 2 == 0)
        half_channels = int(channels/2)

        self.stem_a = nn.Sequential(
            nn.Conv2d(channels, half_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.stem_b = nn.Sequential(
            nn.Conv2d(channels, half_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.k1 = nn.Sequential(
            nn.Conv2d(half_channels, half_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01),
        )

        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(half_channels, half_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01)
        )

        self.k3 = nn.Sequential(
            nn.Conv2d(half_channels, half_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01)
        )

        self.k4 = nn.Sequential(
            nn.Conv2d(half_channels, half_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half_channels, eps=1e-3, momentum=0.01)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        # a path(k1)
        a = self.stem_a(x)
        a_out = self.k1(a)
        a_out = F.relu(a_out)

        # b path(k2,k3,k4)
        b = self.stem_b(x)
        b_out = torch.sigmoid(F.interpolate(self.k2(b), b.shape[2:])+b)
        b_out = torch.mul(self.k3(b), b_out)
        b_out = self.k4(b_out)
        b_out = F.relu(b_out)

        out = self.fusion(torch.cat([a_out, b_out], dim=1))

        out = F.relu(out + x)

        return out
