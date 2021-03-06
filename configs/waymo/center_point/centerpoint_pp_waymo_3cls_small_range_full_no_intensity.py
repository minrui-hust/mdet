from copy import deepcopy as _deepcopy
from mdet.utils.global_config import GCFG

# global config maybe override by command line
batch_size = GCFG['batch_size'] or 2  # different from original, which is 4
num_workers = GCFG['num_workers'] or 4
max_epochs = GCFG['max_epochs'] or 18
lr_scale = GCFG['lr_scale'] or 1.0  # may rescale by gpu number
dataset_root = GCFG['dataset_root'] or '/data/waymo'


# global config
class RawType:
    Vehicle = 1
    Cyclist = 4
    Pedestrian = 2


class Label:
    Vehicle = 0
    Cyclist = 1
    Pedestrian = 2


labels = {
    Label.Vehicle: [RawType.Vehicle],
    Label.Cyclist: [RawType.Cyclist],
    Label.Pedestrian: [RawType.Pedestrian],
}

point_dim = 3

point_range = [-59.52, -59.52, -2, 59.52, 59.52, 4.0]
voxel_size = [0.32, 0.32, 6.0]
voxel_reso = [372, 372, 1]

out_grid_size = [0.32, 0.32]
out_grid_reso = [372, 372]

margin = 1.0

# model config
model_train = dict(
    type='Det3dOneStage',
    voxelization=dict(
        type='PillarVoxelization',
        point_range=point_range,
        voxel_size=voxel_size,
        voxel_reso=voxel_reso,
        max_points=32,
        max_voxels=30000,
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
        pfn_channels=[64, 64, ],
    ),
    backbone2d=dict(
        type='SCConvNet',
        layer_nums=[3, 3, 3],
        layer_strides=[1, 2, 2],
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
        shared_conv_channels=128,
        init_bias=-2.19,
        heads={
            'heatmap': (len(labels), 2),
            'offset': (2, 2),
            'height': (1, 2),
            'size': (3, 2),
            'heading': (2, 2),
            'iou': (1, 2),
        },  # (output_channel, num_conv)
    ),
)

model_eval = _deepcopy(model_train)
model_eval['voxelization']['max_voxels'] = 50000

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
        labels=labels,
        heatmap_encoder=dict(
            type='GaussianBoxHeatmapEncoder',
            grid=out_grid_size[0],
            min_radius=2,
            offset_enable=True,
        ),
    ),
    decode_cfg={
        Label.Vehicle:    dict(pre_num=4096, post_num=512, overlap_thresh=0.8,  iou_gamma=2.0, valid_thresh=0.05),
        Label.Cyclist:    dict(pre_num=4096, post_num=512, overlap_thresh=0.55, iou_gamma=2.0, valid_thresh=0.05),
        Label.Pedestrian: dict(pre_num=4096, post_num=512, overlap_thresh=0.55, iou_gamma=2.0, valid_thresh=0.05),
    },
    loss_cfg=dict(
        head_weight={
            'heatmap': 1.0,
            'offset': 2 * 2.0,
            'height': 1 * 2.0,
            'size': 3 * 2.0,
            'heading': 2 * 2.0,
            'iou': 1.*2.0,
        },
        alpha=2.0,
        beta=4.0,
        full_positive_loss=True,
    ),
)

codec_eval = _deepcopy(codec_train)

codec_infer = _deepcopy(codec_eval)
codec_infer['encode_cfg']['encode_anno'] = False
codec_infer['decode_cfg'] = dict(
    nms_cfg=dict(
        pre_num=1024,
        post_num=128,
        overlap_thresh=0.1,
    ),
    valid_thresh=0.1,
)


# data config
db_sampler = dict(
    type='GroundTruthSamplerV2',
    info_path=f'{dataset_root}/training_info_gt.pkl',
    pcd_loader=dict(type='WaymoObjectNSweepLoader', load_dim=point_dim, nsweep=1),
    interest_types=[RawType.Vehicle, RawType.Cyclist, RawType.Pedestrian],
    filters=[
        dict(type='FilterByNumpoints', min_points_groups=5),
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
        load_opt=dict(load_dim=point_dim, nsweep=1, interest_types=[RawType.Vehicle, RawType.Cyclist, RawType.Pedestrian]),
        transforms=[
            dict(type='PcdObjectSamplerV2', db_sampler=db_sampler, sample_groups={
                RawType.Vehicle: 15,
                RawType.Cyclist: 10,
                RawType.Pedestrian: 10,
            }),
            dict(type='PcdMirrorFlip', mirror_prob=0.5, flip_prob=0.5),
            dict(type='PcdGlobalTransform',
                 rot_range=[-0.78539816, 0.78539816],
                 scale_range=[0.95, 1.05],
                 translation_std=[0.5, 0.5, 0.2]),
            dict(type='PcdRangeFilter', point_range=point_range, margin=margin),
            dict(type='PcdShuffler'),
        ],
        #  filter=dict(type='IntervalDownsampler', interval=5),
    ),
)

dataloader_eval = _deepcopy(dataloader_train)
dataloader_eval['shuffle'] = False
dataloader_eval['dataset']['info_path'] = f'{dataset_root}/validation_info.pkl'
dataloader_eval['dataset']['transforms'] = [
    dict(type='PcdRangeFilter', point_range=point_range, margin=margin),
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
            dict(type='TensorBoardLogger', flush_secs=15),
            dict(type='CSVLogger', flush_logs_every_n_steps=50),
        ],
    ),
    eval=dict(evaluate_min_epoch=max_epochs-1),
    test=dict(),
)
