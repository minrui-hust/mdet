from copy import deepcopy as _deepcopy
from mdet.utils.global_config import GCFG

# global config maybe override by command line
batch_size = GCFG['batch_size'] or 2
max_epochs = GCFG['max_epochs'] or 32
lr_scale = GCFG['lr_scale'] or 1.0
dataset_root = GCFG['dataset_root'] or '/data/waymo'

# global config
types = [('Vehicle', [1])]

point_range = [-75.52, -75.52, -2, 75.52, 75.52, 4.0]
voxel_size = [0.32, 0.32, 6.0]
voxel_reso = [472, 472, 1]

out_grid_size = [0.64, 0.64]
out_grid_reso = [236, 236]

point_dim = 5

# model config
model_train = dict(
    type='Det3dOneStage',
    voxelization=dict(
        type='PillarVoxelization',
        point_range=point_range,
        voxel_size=voxel_size,
        voxel_reso=voxel_reso,
        max_points=32,
        max_voxels=32000,
    ),
    backbone3d=dict(
        type='PillarFeatureNet',
        pillar_feat=[
            ('position', 3),
            ('attribute', point_dim-3),
            ('center_offset', 2),
            ('mean_offset', 3),
            #  ('distance', 1),
        ],
        voxel_reso=voxel_reso,
        voxel_size=voxel_size,
        point_range=point_range,
        pfn_channels=[64, ],
    ),
    backbone2d=dict(
        type='SECOND',
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
        in_channels=64,
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
    ),
    head=dict(
        type='CenterHead',
        in_channels=3 * 128,
        shared_conv_channels=64,
        init_bias=-2.19,
        heads={
            'heatmap': (len(types), 2),
            'offset': (2, 2),
            'height': (1, 2),
            'size': (3, 2),
            'heading': (2, 2),
        },  # (output_channel, num_conv)
    ),
)

model_eval = _deepcopy(model_train)

model_infer = _deepcopy(model_train)


# codecs config
codec_train = dict(
    type='CenterPointCodec',
    encode_cfg=dict(
        encode_data=True,
        encode_anno=True,
        point_range=point_range,
        grid_size=out_grid_size,
        grid_reso=out_grid_reso,
        min_gaussian_radius=2,
        min_gaussian_overlap=0.1,
    ),
    decode_cfg=dict(
        nms_cfg=dict(
            pre_num=1024,
            post_num=256,
            overlap_thresh=0.1,
        ),
    ),
    loss_cfg=dict(
        head_weight={
            'heatmap': 2.0,
            'offset': 2.0,
            'height': 1.0,
            'size': 3.0,
            'heading': 2.0,
        },
        alpha=4.0,
        beta=2.0,
    ),
)

codec_eval = _deepcopy(codec_train)

codec_infer = _deepcopy(codec_eval)
codec_infer['encode_cfg']['encode_anno'] = False


# data config
dataloader_train = dict(
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    dataset=dict(
        type='WaymoDet3dDataset',
        info_path=f'{dataset_root}/training_info.pkl',
        load_opt=dict(load_dim=point_dim, nsweep=1, types=types,),
        transforms=[
            dict(type='PcdIntensityNormlizer'),
            dict(type='PcdRangeFilter', point_range=point_range),
            dict(type='PcdShuffler'),
        ],
        filter=dict(type='IntervalDownsampler', interval=5),
    ),
)

dataloader_eval = _deepcopy(dataloader_train)
dataloader_eval['shuffle'] = False
dataloader_eval['dataset']['info_path'] = f'{dataset_root}/validation_info.pkl'
dataloader_eval['dataset']['transforms'] = [
    dict(type='PcdIntensityNormlizer'),
    dict(type='PcdRangeFilter', point_range=point_range),
    dict(type='PcdShuffler'),
]
dataloader_eval['dataset']['filter'] = None

dataloader_infer = _deepcopy(dataloader_eval)

# collect config
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

fit = dict(
    max_epochs=max_epochs,
    optimizer=dict(
        type='AdamW',
        weight_decay=0.01,
        betas=(0.9, 0.99),
    ),
    scheduler=dict(
        type='OneCycleLR',
        max_lr=0.001 / 16 * batch_size * lr_scale,
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
    eval=dict(output_folder=None, evaluate=False),
    test=dict(),
)
