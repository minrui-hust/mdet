import torch
from mdet.utils.factory import FI
import mdet.model

# global config
############################
model_dim = 128
pos_dim = 64
dist_dim = 128
lane_enc_dim = 64
agent_dim = 4
num_queries = 6
dropout = 0.0
num_heads = 2
agent_layers = 2
lane_layers = 2
social_layers = 2
pred_win = 30


# model config
############################
model = dict(
    type='MMTrans',
    hparam=dict(
        model_dim=model_dim,
        pos_dim=pos_dim,
        dist_dim=dist_dim,
        lane_enc_dim=lane_enc_dim,
        agent_dim=agent_dim,
        dropout=dropout,
        num_queries=num_queries,
    ),
    agent_enc=dict(
        type='TransformerEncoder',
        layer_num=agent_layers,
        layer_cfg=dict(
            type='TransformerEncoderLayer',
            atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    agent_dec=dict(
        type='TransformerDecoder',
        layer_num=agent_layers,
        layer_cfg=dict(
            type='TransformerDecoderLayer',
            self_atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            cross_atten_cfg=dict(
                type='MultiHeadCrossAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    lane_net=dict(
        type='LaneNet',
        in_channels=7,
        hidden_unit=64,
        layer_num=2,
    ),
    lane_enc=dict(
        type='TransformerEncoder',
        layer_num=lane_layers,
        layer_cfg=dict(
            type='TransformerEncoderLayer',
            atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    lane_dec=dict(
        type='TransformerDecoder',
        layer_num=lane_layers,
        layer_cfg=dict(
            type='TransformerDecoderLayer',
            self_atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            cross_atten_cfg=dict(
                type='MultiHeadCrossAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    social_enc=dict(
        type='TransformerEncoder',
        layer_num=social_layers,
        layer_cfg=dict(
            type='TransformerEncoderLayer',
            atten_cfg=dict(
                type='MultiHeadSelfAtten',
                embed_dim=model_dim,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ff_cfg=dict(
                type='MLP',
                in_channels=model_dim,
                hidden_channels=model_dim*2,
                out_channels=model_dim,
                norm_cfg=None,
            ),
            norm_cfg=dict(
                type='LayerNorm',
                normalized_shape=model_dim,
            ),
        ),
    ),
    head=dict(
        type='MLPHead',
        in_channels=model_dim*2,
        heads={
            'traj': (model_dim, pred_win*2),
            'score': (model_dim, 1),
        },
    ),
)

model = FI.create(model)

batch = dict(
    input=dict(
        agent=torch.rand((2, 4, 19, 4), dtype=torch.float32),
        lane=torch.rand((2, 8, 9, 7), dtype=torch.float32),
        pos=torch.rand((2, 4, 2), dtype=torch.float32),
        agent_num=torch.full((2,), 4, dtype=torch.int32),
        lane_num=torch.full((2,), 8, dtype=torch.int32),
    )
)

out = model(batch)
print(out['traj'][0][0])
print(out['score'][0][0])
