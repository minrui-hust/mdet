import torch
import torch.nn as nn
import torch.nn.functional as F

from mdet.utils.factory import FI


@FI.register
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=None,
                 linear_cfg=dict(type='Linear'),
                 norm_cfg=dict(type='LayerNorm'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        if out_channels is None:
            out_channels = hidden_channels

        self.lin0 = FI.create(linear_cfg, in_channels, hidden_channels)
        self.norm = FI.create(norm_cfg, hidden_channels)
        self.act = FI.create(act_cfg)
        self.lin1 = FI.create(linear_cfg, hidden_channels, out_channels)

    def forward(self, x):
        lin0_out = self.lin0(x)
        if self.norm is not None:
            lin0_out = self.norm(lin0_out)
        if self.act is not None:
            lin0_out = self.act(lin0_out)
        return self.lin1(lin0_out)


@FI.register
class ResBlock(nn.Module):
    def __init__(self, residual_cfg,
                 norm_cfg=dict(type='LayerNorm', normalized_shape=64),
                 act_cfg=dict(type='ReLU', inplace=True),
                 sampler=None):
        super().__init__()

        self.residual = FI.create(residual_cfg)
        self.norm = FI.create(norm_cfg)
        self.act = FI.create(act_cfg)
        self.sampler = sampler

    def forward(self, x, *args, **kwargs):
        identity = x
        if self.sampler:
            identity = self.sampler(identity)

        out = self.norm(identity + self.residual(x, *args, **kwargs))

        if self.act:
            out = self.act(out)

        return out


@FI.register
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 atten_cfg=dict(type='MultiHeadSelfAtten'),
                 ff_cfg=dict(type='MLP'),
                 norm_cfg=dict(type='LayerNorm')):
        super().__init__()

        self.atten_res_block = ResBlock(atten_cfg, norm_cfg, act_cfg=None)
        self.ff_res_block = ResBlock(ff_cfg, norm_cfg, act_cfg=None)

    def forward(self, x, **args):
        att_out = self.atten_res_block(x, **args)
        ff_out = self.ff_res_block(att_out)
        return ff_out


@FI.register
class TransformerDecoderLayer(nn.Module):
    def __init__(self, self_atten_cfg=dict(type='MultiHeadSelfAtten'),
                 cross_atten_cfg=dict(type='MultiHeadAtten'),
                 ff_cfg=dict(type='MLP'),
                 norm_cfg=dict(type='LayerNorm')):
        super().__init__()

        self.self_atten_res_block = ResBlock(
            self_atten_cfg, norm_cfg, act_cfg=None)
        self.cross_atten_res_block = ResBlock(
            cross_atten_cfg, norm_cfg, act_cfg=None)
        self.ff_res_block = ResBlock(ff_cfg, norm_cfg, act_cfg=None)

    def forward(self, x, y, **args):
        self_atten_out = self.self_atten_res_block(x)
        cross_atten_out = self.cross_atten_res_block(x, y, **args)
        ff_out = self.ff_res_block(cross_atten_out)
        return ff_out


@FI.register
class TransformerEncoder(nn.Module):
    def __init__(self, layer_cfg, layer_num=1):
        super().__init__()

        self.layer_list = nn.ModuleList(
            [FI.create(layer_cfg) for _ in range(layer_num)])

    def forward(self, x, **args):
        for layer in self.layer_list:
            x = layer(x, **args)
        return x


@FI.register
class TransformerDecoder(nn.Module):
    def __init__(self, layer_cfg, layer_num=1):
        super().__init__()

        self.layer_list = nn.ModuleList(
            [FI.create(layer_cfg) for _ in range(layer_num)])

    def forward(self, x, y, **args):
        for layer in self.layer_list:
            x = layer(x, y, **args)
        return x
