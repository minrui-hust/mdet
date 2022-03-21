import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from mai.model import BaseModule
from mai.utils import FI


@FI.register
class DetrHead(BaseModule):
    def __init__(self, num_proposals=128, proposal_dim=64, feature_dims=[], block_num=2, num_sa_heads=4, dropout=0.0, heads={}, init_bias=-2.19):
        super().__init__()

        self.proposals = Parameter(data=torch.empty(
            (num_proposals, proposal_dim), dtype=torch.float32))
        nn.init.uniform_(self.proposals)

        self.detr_blocks = nn.ModuleList([
            DetrBlock(proposal_dim, feature_dims, num_sa_heads, dropout) for _ in range(block_num)
        ])

        self.head = SepHead(proposal_dim, heads, init_bias=init_bias)

    def forward_train(self, feats):
        x = self.proposals.unsqueeze(0).expand(feats[0].shape[0], -1, -1)
        for detr_block in self.detr_blocks:
            x = detr_block(x, feats)
        out_dict = self.head(x)
        return out_dict


class DetrBlock(nn.Module):
    def __init__(self, proposal_dim, feature_dims, num_heads, dropout):
        super().__init__()

        self.layers = nn.ModuleList([
            DetrLayer(proposal_dim, feature_dim, num_heads, dropout) for feature_dim in feature_dims
        ])

    def forward(self, x, feats):
        assert (len(feats) == len(self.layers))
        for i in range(len(feats)):
            x = self.layers[-i-1](x, feats[-i-1])
        return x


class DetrLayer(nn.Module):
    def __init__(self, proposal_dim, feature_dim, num_heads=4, dropout=0.0):
        super().__init__()

        self.pos_dec = nn.Sequential(
            nn.Linear(proposal_dim, proposal_dim),
            nn.LayerNorm(proposal_dim),
            nn.ReLU(),
            nn.Linear(proposal_dim, 2),
        )

        self.ff = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, proposal_dim),
        )

        self.sa = MultiHeadSelfAttention(
            embed_dim=proposal_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x, feat):
        r'''
        x: proposal feature, shape [B, N, F]
        feat: feature to fuse, shape [B, C, H, W]
        '''

        # pos shape [B, 1, N, 2]
        pos = self.pos_dec(x).unsqueeze(1)
        #  print(pos)

        sample = F.grid_sample(
            feat, pos, align_corners=True).squeeze(2).transpose(1, 2)

        #  x = self.sa(x + self.ff(sample))
        x = x + self.ff(sample)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.0):
        super().__init__()

        self.sa = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm_sa = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        sa_out = self.norm_sa(self.sa(x, x, x)[0] + x)
        lin_out = self.ff_norm(self.ff(sa_out)+sa_out)
        return lin_out


class SepHead(nn.Module):
    def __init__(self, in_channels, heads={}, init_bias=-2.19):
        super().__init__()

        self.heads = nn.ModuleDict()
        for head_name, channels in heads.items():
            head = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, channels),
            )

            if head_name == 'cls':
                head[-1].bias.data.fill_(init_bias)

            self.heads[head_name] = head

    def forward(self, x):
        return {head_name: head(x) for head_name, head in self.heads.items()}
