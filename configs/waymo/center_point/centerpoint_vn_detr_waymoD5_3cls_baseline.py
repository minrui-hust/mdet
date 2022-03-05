from copy import deepcopy as _deepcopy
from mdet.utils.global_config import GCFG

# global config maybe override by command line
batch_size = GCFG['batch_size'] or 2  # different from original, which is 4
num_workers = GCFG['num_workers'] or 4
max_epochs = GCFG['max_epochs'] or 36
lr_scale = GCFG['lr_scale'] or 1.0  # may rescale by gpu number
dataset_root = GCFG['dataset_root'] or '/data/waymo'


# global config
class RawType:
    Vehicle = 1
    Cyclist = 4
    Pedestrian = 2


class Label:
    Null = 0
    Vehicle = 1
    Cyclist = 2
    Pedestrian = 3


labels = {
    Label.Vehicle: [RawType.Vehicle],
    Label.Cyclist: [RawType.Cyclist],
    Label.Pedestrian: [RawType.Pedestrian],
}

point_dim = 5

point_range = [-75.2, -75.2, -2, 75.2, 75.2, 4.0]
voxel_size = [0.1, 0.1, 0.15]
voxel_reso = [1504, 1504, 40]

# 8x times downsample
out_grid_size = [0.8, 0.8]
out_grid_reso = [188, 188]

margin = 1.0


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
    neck=dict(type='DummyNeck'),
    head=dict(
        type='DetrHead',
        num_proposals=1024,
        proposal_dim=128,
        feature_dims=[128, 256],
        block_num=3,
        num_sa_heads=4,
        dropout=0.0,
        init_bias=-2.19,
        heads={
            'cls': len(labels)+1,
            'pos': 2,
            'height': 1,
            'size': 3,
            'heading': 2,
        },  # (output_channel)
    ),
)

# model for evaluation
model_eval = _deepcopy(model_train)
model_eval['voxelization']['max_voxels'] = 300000

# model for inference(export)
model_infer = _deepcopy(model_eval)


# codecs config
codec_train = dict(
    type='DetrCodec',
    encode_cfg=dict(
        encode_data=True,
        encode_anno=True,
        point_range=point_range,
        grid_size=out_grid_size,
        grid_reso=out_grid_reso,
        labels=labels,
    ),
    decode_cfg=dict(
    ),
    loss_cfg=dict(
        head_weight={
            'cls': 1.0,
            'pos': 1.0,
            'height': 0,  # 1.0,
            'size': 0,  # 1.0,
            'heading': 0,  # 1.0,
        },
    ),
)

codec_eval = _deepcopy(codec_train)

codec_infer = _deepcopy(codec_eval)


# data config
db_sampler = dict(
    type='GroundTruthSamplerV2',
    info_path=f'{dataset_root}/training_info_gt.pkl',
    pcd_loader=dict(type='WaymoObjectNSweepLoader', load_dim=5, nsweep=1),
    interest_types=[RawType.Vehicle, RawType.Cyclist, RawType.Pedestrian],
    filters=[
        dict(type='FilterByNumpointsV2', min_points_groups=5),
        dict(type='FilterByRange', range=point_range),
    ],
)


dataloader_train = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=False,
    dataset=dict(
        type='WaymoDet3dDataset',
        info_path=f'{dataset_root}/training_info.pkl',
        load_opt=dict(load_dim=point_dim, nsweep=1, interest_types=[
                      RawType.Vehicle, RawType.Cyclist, RawType.Pedestrian],),
        transforms=[
            dict(type='PointNumFilterV2', groups=1),
            #  dict(type='PcdObjectSamplerV2', db_sampler=db_sampler, sample_groups={
            #      RawType.Vehicle: 15,
            #      RawType.Cyclist: 10,
            #      RawType.Pedestrian: 10,
            #  }),
            #  dict(type='PcdMirrorFlip', mirror_prob=0.5, flip_prob=0.5),
            #  dict(type='PcdGlobalTransform',
            #       rot_range=[-0.78539816, 0.78539816],
            #       scale_range=[0.95, 1.05],
            #       translation_std=[0.5, 0.5, 0]),
            dict(type='PcdRangeFilter', point_range=point_range, margin=margin),
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
    dict(type='PcdRangeFilter', point_range=point_range, margin=margin),
    dict(type='PcdIntensityNormlizer'),
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
        max_lr=0.003,  # (0.003 / 16) * batch_size * lr_scale,
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
