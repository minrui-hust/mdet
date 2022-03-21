from copy import deepcopy as _deepcopy
from mdet.utils.global_config import GCFG

# global config maybe override by command line
batch_size = GCFG['batch_size'] or 2  # different from original, which is 4
max_epochs = GCFG['max_epochs'] or 36
lr_scale = GCFG['lr_scale'] or 1.0  # may rescale by gpu number
dataset_root = GCFG['dataset_root'] or '/data/waymo'

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


# model config
############################
model_train = dict(
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
    lane_net=dict(
        type='LaneNet',
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
    head=dict(),
)

model_eval = _deepcopy(model_train)

model_infer = _deepcopy(model_train)


# codecs config
############################
codec_train = dict(
)

codec_eval = _deepcopy(codec_train)

codec_infer = _deepcopy(codec_eval)
codec_infer['encode_cfg']['encode_anno'] = False


dataloader_train = dict(
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    dataset=dict(
    ),
)

dataloader_eval = _deepcopy(dataloader_train)
dataloader_eval['shuffle'] = False
dataloader_eval['dataset']['info_path'] = f'{dataset_root}/validation_info.pkl'
dataloader_eval['dataset']['transforms'] = []
dataloader_eval['dataset']['filter'] = None

dataloader_infer = _deepcopy(dataloader_eval)


# fit config
############################
fit = dict(
    max_epochs=max_epochs,
    optimizer=dict(
        type='AdamW',
        weight_decay=0.01,
        betas=(0.9, 0.99),
    ),
    scheduler=dict(
        type='OneCycleLR',
        max_lr=0.003 / 4 * batch_size * lr_scale,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=10.0,
        pct_start=0.4,
    ),
    grad_clip=dict(type='norm', value=35),
)


runtime = dict(
    train=dict(
        logger=[
            dict(type='TensorBoardLogger',),
            dict(type='CSVLogger',),
        ],
    ),
    eval=dict(evaluate_min_epoch=max_epochs-1),
    test=dict(),
)


# collect config
#################################
model = dict(
    train=model_train,
    eval=model_eval,
    infer=model_infer,
)

codec = dict(
    train=codec_train,
    eval=codec_eval,
    infer=codec_infer,
)

data = dict(
    train=dataloader_train,
    eval=dataloader_eval,
    infer=dataloader_infer,
)
