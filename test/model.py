from mdet.utils.factory import FI
import mdet.data
import mdet.model
import torch

import open3d as o3d
from mdet.ops.voxelization import Voxelize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data.dataloader import DataLoader


types = [('Pedestrian', [2]), ('Cyclist', [4]), ('Vehicle', [1])]

point_range = [-64, -64, -5, 64, 64, 5]
voxel_size = [0.32, 0.32, 10]
voxel_reso = [400, 400, 1]

out_grid_size = [0.64, 0.64]
out_grid_reso = [200, 200]

codec = dict(
    type='CenterPointCodec',
    point_range=point_range,
    grid_size=out_grid_size,
    grid_reso=out_grid_reso,
    min_gaussian_radius=2,
    min_gaussian_overlap=0.5,
)

dataset_config = dict(
    type='WaymoDet3dDataset',
    info_path='/data/tmp/waymo/training_info.pkl',
    load_opt=dict(
        load_dim=5,
        num_sweeps=1,
        types=types,
    ),
    transforms=[
        dict(type='RangeFilter', point_range=point_range),
        codec,
    ],
)
dataset = FI.create(dataset_config)

collator_config = dict(
    type='SimpleCollator',
    rules={
        '.anno.offset': dict(type='cat'),
        '.anno.height': dict(type='cat'),
        '.anno.size': dict(type='cat'),
        '.anno.heading': dict(type='cat'),
        '.anno.heatmap': dict(type='stack'),
        '.anno.types': dict(type='cat'),
        '.anno.positive_indices': dict(
            type='cat',
            dim=0,
            pad=dict(pad_width=((0, 0), (1, 0)),),
            inc_func=lambda x: np.array([1, 0, 0], dtype=np.int32),
        ),
        '.type_name': dict(type='unique'),
    },
)
collator = FI.create(collator_config)

dataloader = DataLoader(dataset, 2, collate_fn=collator)


model_config = dict(
    type='Det3dOneStage',
    voxelization=dict(type='PillarVoxelization',
                      voxel_size=voxel_size,
                      point_range=point_range,
                      max_points=32,
                      max_voxels=15000,
                      ),
    backbone3d=dict(type='PillarFeatureNet',
                    pillar_feat=[('position', 3),
                                 ('attribute', 2),
                                 ('center_offset', 2),
                                 ('mean_offset', 3),
                                 ('distance', 1)],
                    voxel_size=voxel_size,
                    point_range=point_range,
                    pfn_channels=(64,),
                    ),
    backbone2d=dict(type='SECOND',
                    in_channels=64,
                    out_channels=[128, 128, 256],
                    layer_nums=[3, 5, 5],
                    layer_strides=[2, 2, 2],
                    ),
    neck=dict(type='SECONDFPN',
              in_channels=[128, 128, 256],
              out_channels=[128, 128, 128],
              upsample_strides=[1, 2, 4],
              ),
    head=dict(type='CenterHead',
              in_channels=3*128,
              shared_conv_channels=64,
              heads={'heatmap': (3, 2),
                     'offset': (2, 2),
                     'height': (1, 2),
                     'size': (3, 2),
                     'heading': (2, 2)},  # (output_channel, num_conv)
              init_bias=-2.20,
              ),
    post_process=dict(
        type='CenterPostProcess',
        codec=codec,  # use the same one as dataset transforms
        loss_cfg=dict(
            head_weight={
                'heatmap': 1.0,
                'offset': 1.0,
                'height': 1.0,
                'size': 1.0,
                'heading': 1.0,
            },
            alpha=4.0,
            beta=2.0,
        ),
        nms_cfg=dict(
            pre_num=512,
            post_num=128,
            overlap_thresh=0.1,
        ),
    )
)
model = FI.create(model_config)
model.set_eval()
model.cuda()


for batch in dataloader:
    batch.to('cuda')
    res = model(batch)
    print(res)
    break
