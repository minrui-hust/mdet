import torch
from copy import deepcopy


# global config
types = [('Vehicle', [1])]

point_range = [-75.2, -75.2, -2, 75.2, 75.2, 4.0]
voxel_size = [0.1, 0.1, 0.15]
voxel_reso = [1504, 1504, 40]

# 8x times downsample
out_grid_size = [0.8, 0.8]
out_grid_reso = [188, 188]

batch_size = 2
max_epochs = 96

# model config
# model for training
model_train = dict(
    type='Det3dOneStage',
    voxelization=dict(
        type='PillarVoxelization',
        point_range=point_range,
        voxel_size=voxel_size,
        voxel_reso=voxel_reso,
        max_points=5,
        max_voxels=150000,
        reduce_type='mean',
    ),
    backbone3d=dict(
        type='SparseResNetFHD',
        in_channels=5,
        # out_channels=256,
    ),
    backbone2d=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256],
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
    ),
    head=dict(
        type='CenterHead',
        in_channels=2 * 256,
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

# model for evaluation
model_eval = deepcopy(model_train)

# model for inference(export)
model_infer = deepcopy(model_train)


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
            overlap_thresh=0.01,
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

codec_eval = deepcopy(codec_train)

codec_infer = deepcopy(codec_train)


# data config
dataloader_train = dict(
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    dataset=dict(
        type='WaymoDet3dDataset',
        info_path='/data/tmp/waymo/training_info.pkl',
        load_opt=dict(
            load_dim=5,
            num_sweeps=1,
            types=types,
        ),
        transforms=[
            dict(type='PcdRangeFilter', point_range=point_range),
            dict(type='PcdIntensityNormlizer'),
        ],
    ),
)

dataloader_eval = deepcopy(dataloader_train)
dataloader_eval['shuffle'] = False
dataloader_eval['dataset']['info_path'] = '/data/tmp/waymo/validation_info.pkl'
#  dataloader_val['dataset']['info_path'] = '/data/tmp/waymo/training_info.pkl'

dataloader_infer = deepcopy(dataloader_train)
dataloader_infer['shuffle'] = False
dataloader_infer['dataset']['info_path'] = '/data/tmp/waymo/validation_info.pkl'


# collect config
model = dict(
    train=model_train,
    eval=model_eval,
    infer=model_infer,
)

codec = dict(
    train=codec_train,
    eval=codec_train,
    infer=codec_train,
)

data = dict(
    train=dataloader_train,
    eval=dataloader_eval,
    infer=dataloader_infer,
)


fit = dict(
    max_epochs=max_epochs,
    optimizer=dict(
        type='Adam',
        betas=(0.9, 0.99),
    ),
    scheduler=dict(
        type='OneCycleLR',
        max_lr=3e-4 / 2,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=10.0,
        pct_start=0.4,
    ),
)

runtime = dict(
    train=dict(
        logger=[
            dict(type='TensorBoardLogger',),
            #  dict(type='CSVLogger',),
        ],
    ),
    eval=dict(output_folder=None, evaluate=False),
    test=dict(),
)
