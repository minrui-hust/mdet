from mai.model import BaseModule
import mai.model.uitls.init as init
from mai.utils import FI
import torch
import torch.nn as nn
from torch.nn import init


@FI.register
class CenterHead(BaseModule):
    def __init__(self,
                 in_channels=128,
                 shared_conv_channels=64,
                 heads={},
                 init_bias=-2.19
                 ):
        super().__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, shared_conv_channels,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(shared_conv_channels),
            nn.ReLU(inplace=True)
        )

        self.head = SepHead(shared_conv_channels, heads,
                            bn=True, init_bias=init_bias, final_kernel=3)

    def forward_train(self, x):
        x = self.shared_conv(x)
        out_dict = self.head(x)

        return out_dict


class SepHead(BaseModule):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 bn=False,
                 init_bias=-2.19
                 ):
        super().__init__()

        self.heads = nn.ModuleDict()
        for head_name, (channels, num_conv) in heads.items():

            fc = []
            for i in range(num_conv-1):
                fc.append(nn.Conv2d(in_channels, head_conv,
                                    kernel_size=final_kernel, stride=1,
                                    padding=final_kernel // 2, bias=True))
                if bn:
                    fc.append(nn.BatchNorm2d(head_conv))
                fc.append(nn.ReLU())

            final_bias = not head_name == 'iou'  # iou no bias
            fc.append(nn.Conv2d(head_conv, channels,
                                kernel_size=final_kernel, stride=1,
                                padding=final_kernel // 2, bias=final_bias))
            fc = nn.Sequential(*fc)

            # special init for heatmap head
            if head_name in ['heatmap', 'keypoint_map']:
                fc[-1].bias.data.fill_(init_bias)
            else:  # special init for conv2d of non heatmap heads
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        init.kaiming_init(m)

            self.heads[head_name] = fc

    def forward(self, x):
        return {head_name: head(x) for head_name, head in self.heads.items()}
