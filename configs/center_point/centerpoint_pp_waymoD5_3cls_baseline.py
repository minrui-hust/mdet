import torch
from copy import deepcopy


# global config
types = [('Vehicle', [1]), ('Cyclist', [4]), ('Pedestrian', [2])]

point_range = [-75.52, -75.52, -2, 75.52, 75.52, 4.0]
voxel_size = [0.32, 0.32, 6.0]
voxel_reso = [472, 472, 1]

out_grid_size = [0.64, 0.64]
out_grid_reso = [236, 236]

batch_size = 2
max_epochs = 36
lr_scale = 1.0
dataset_root = '/data/tmp/waymo'

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
            ('attribute', 2),
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

model_eval = deepcopy(model_train)

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

codec_eval = deepcopy(codec_train)

codec_infer = deepcopy(codec_train)
codec_infer['encode_cfg']['encode_anno'] = False


# data config
db_sampler = dict(
    type='GroundTruthSampler',
    info_path=f'{dataset_root}/training_info_gt.pkl',
    sample_groups={'Pedestrian': 10, 'Cyclist': 10, 'Vehicle': 15},
    labels=types,
    pcd_loader=dict(type='WaymoNSweepLoader', load_dim=5, nsweep=1),
    filter=dict(type='FilterByNumpoints', min_num_points=10), # TODO
)

dataloader_train = dict(
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    dataset=dict(
        type='WaymoDet3dDataset',
        info_path=f'{dataset_root}/training_info.pkl',
        load_opt=dict(load_dim=5, nsweep=1, types=types,),
        transforms=[
            dict(type='PcdIntensityNormlizer'),
            dict(type='PcdObjectSampler', db_sampler=db_sampler),
            dict(type='PcdGlobalTransform', rot_range=[-0.78539816, 0.78539816], scale_range=[0.95, 1.05]),
            dict(type='PcdRangeFilter', point_range=point_range),
            dict(type='PcdShuffler'),
        ],
        filter=dict(type='IntervalDownsampler', interval=5),
    ),
)

dataloader_eval = deepcopy(dataloader_train)
dataloader_eval['shuffle'] = False
dataloader_eval['dataset']['info_path'] = f'{dataset_root}/validation_info.pkl'

dataloader_infer = deepcopy(dataloader_train)
dataloader_infer['shuffle'] = False
dataloader_infer['dataset']['info_path'] = f'{dataset_root}/validation_info.pkl'

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
        type='Adam',
        weight_decay=0.01,
        betas=(0.9, 0.99),
        lr=0.003 / 16 * batch_size * lr_scale,
    ),
    scheduler=dict(
        type='OneCycleLR',
        max_lr=0.003 / 16 * batch_size * lr_scale,
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
