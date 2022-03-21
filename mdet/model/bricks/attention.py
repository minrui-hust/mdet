import torch
import torch.nn as nn
import torch.nn.functional as F

from mdet.utils.factory import FI


@FI.register
class MultiHeadSelfAtten(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()

        self.atten = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout,
                                           batch_first=batch_first)

    def forward(self, x, mask=None):
        return self.atten(x, x, x, key_padding_mask=mask, need_weights=False)[0]


@FI.register
class MultiHeadCrossAtten(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()

        self.atten = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout,
                                           batch_first=batch_first)

    def forward(self, x, y, mask=None):
        return self.atten(x, y, y, key_padding_mask=mask, need_weights=False)[0]
