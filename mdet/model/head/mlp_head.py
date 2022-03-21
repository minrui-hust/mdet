import torch
import torch.nn as nn
import torch.nn.functional as F

from mai.model import BaseModule
from mai.utils import FI


@FI.register
class MLPHead(BaseModule):
    def __init__(self, in_channels, heads={}):
        super().__init__()

        self.heads = nn.ModuleDict()
        for name, head in heads.items():
            self.heads[name] = FI.create(
                dict(type='MLP', in_channels=in_channels, hidden_channels=head[0], out_channels=head[1]))

    def forward(self, x):
        return {name: head(x) for name, head in self.heads.items()}
