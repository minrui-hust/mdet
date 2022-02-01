from copy import deepcopy as _deepcopy
from mdet.utils.global_config import GCFG

# global config maybe override by command line
batch_size = GCFG['batch_size'] or 2  # different from original, which is 4
max_epochs = GCFG['max_epochs'] or 36
lr_scale = GCFG['lr_scale'] or 1.0  # may rescale by gpu number
dataset_root = GCFG['dataset_root'] or '/data/waymo'

# global config
labels = [('Vehicle', [1]), ('Cyclist', [4]), ('Pedestrian', [2])]

point_range = [-75.2, -75.2, -2, 75.2, 75.2, 4.0]
voxel_size = [0.1, 0.1, 0.15]
voxel_reso = [1504, 1504, 40]

# 8x times downsample
out_grid_size = [0.8, 0.8]
out_grid_reso = [188, 188]

point_dim = 5

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
        in_channels=point_dim,
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
            'heatmap': (len(labels), 2),
            'offset': (2, 2),
            'height': (1, 2),
            'size': (3, 2),
            'heading': (2, 2),
        },  # (output_channel, num_conv)
    ),
)

# model for evaluation
model_eval = _deepcopy(model_train)

# model for inference(export)
model_infer = _deepcopy(model_eval)


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
        labels=labels,
    ),
    decode_cfg=dict(
        nms_cfg=dict(
            pre_num=4096,
            post_num=512,
            overlap_thresh=0.7,
        ),
        valid_thresh=0.1,
    ),
    loss_cfg=dict(
        head_weight={
            'heatmap': 1.0,
            'offset': 2 * 2.0,
            'height': 1 * 2.0,
            'size': 3 * 2.0,
            'heading': 2 * 2.0,
        },
        alpha=2.0,
        beta=4.0,
    ),
)

codec_eval = _deepcopy(codec_train)

codec_infer = _deepcopy(codec_eval)


# data config
db_sampler = dict(
    type='GroundTruthSampler',
    info_path=f'{dataset_root}/training_info_gt.pkl',
    sample_groups=[('Vehicle', 15), ('Pedestrian', 10), ('Cyclist', 10)],
    labels=labels,
    pcd_loader=dict(type='WaymoObjectNSweepLoader', load_dim=5, nsweep=1),
    filters=[dict(type='FilterByNumpoints', min_num_points=5), ],
)

dataloader_train = dict(
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    dataset=dict(
        type='WaymoDet3dDataset',
        info_path=f'{dataset_root}/training_info.pkl',
        load_opt=dict(load_dim=point_dim, nsweep=1, labels=labels,),
        transforms=[
            dict(type='PcdObjectSampler', db_sampler=db_sampler),
            dict(type='PcdMirrorFlip', mirror_prob=0.5, flip_prob=0.5),
            dict(type='PcdGlobalTransform',
                 rot_range=[-0.78539816, 0.78539816],
                 scale_range=[0.95, 1.05],
                 translation_std=[0.5, 0.5, 0]),
            dict(type='PcdRangeFilter', point_range=point_range),
            dict(type='PcdIntensityNormlizer'),
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
        type='AdamW',
        weight_decay=0.01,
        betas=(0.9, 0.99),
    ),
    scheduler=dict(
        type='OneCycleLR',
        max_lr=(0.003 / 16) * batch_size * lr_scale,
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
