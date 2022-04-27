from mai.utils import FI
from mai.model.bricks.normalize import FrozenBatchNorm2d
import torch
from torch.cuda.amp.autocast_mode import autocast

import mdet.model

# pcd config
point_dim = 4

point_range = [-75.2, -75.2, -2, 75.2, 75.2, 4.0]
voxel_size = [0.2, 0.2, 0.3]
voxel_reso = [752, 752, 20]

# config
query_grid_size = [0.8, 0.8, 1.5]
query_grid_reso = [188, 188, 4]

# img config
in_w = 256
in_h = 128
key_shapes = [
    #  (int(in_w/8),  int(in_h/8)),
    (int(in_w/16), int(in_h/16)),
    (int(in_w/32), int(in_h/32)),
]
n_cams = 2

# model global config
d_fusion = 128
n_heads = 2
dropout = 0
deform_sample_points = 2


model_config = dict(
    type='BevFusionNet',
    pcd_encoder=dict(
        type='PcdEncoder',
        voxelization=dict(
            type='PillarVoxelization',
            point_range=point_range,
            voxel_size=voxel_size,
            voxel_reso=voxel_reso,
            max_points=20,
            max_voxels=32000,
        ),
        backbone3d=dict(
            type='PillarFeatureNet',
            pillar_feat=[
                ('position', 3),
                ('attribute', point_dim-3),
                ('center_offset', 2),
                ('mean_offset', 3),
            ],
            point_range=point_range,
            voxel_size=voxel_size,
            voxel_reso=voxel_reso,
            pfn_channels=[64, 64],
        ),
        backbone2d=dict(
            type='SECOND',
            layer_nums=[3, 3],
            layer_strides=[2, 2],
            in_channels=64,
            out_channels=[int(d_fusion/2), d_fusion],
        ),
    ),
    img_encoder=dict(
        type='ImgEncoder',
        backbone=dict(
            type='ModelIntermediateLayers',
            model=dict(
                type='resnet34',
                pretrained=True,
                norm_layer=FrozenBatchNorm2d,
            ),
            #  interm_layers=dict(layer2='0', layer3='1', layer4='2'),
            interm_layers=dict(layer3='0', layer4='1'),
        ),
        neck=dict(
            type='SECONDFPN',
            #  in_channels=[128, 256, 512],
            #  out_channels=[d_fusion, d_fusion, d_fusion],
            #  upsample_strides=[1, 1, 1],
            in_channels=[256, 512],
            out_channels=[d_fusion, d_fusion],
            upsample_strides=[1, 1],
            cat_output_list=False,
        ),
    ),
    fusion=dict(
        type='BevTransformerFusion',
        transformer=dict(
            type='TransformerDecoder',
            layer_num=2,
            layer_cfg=dict(
                type='TransformerDecoderLayer',
                self_atten_cfg=dict(
                    type='BevDeformSelfAttn',
                    d_model=d_fusion,
                    n_heads=n_heads,
                    n_points=deform_sample_points,
                    grid_reso=query_grid_reso,
                ),
                cross_atten_cfg=dict(
                    type='BevDeformCrossAttn',
                    d_model=d_fusion,
                    n_levels=len(key_shapes),
                    n_heads=n_heads,
                    n_points=deform_sample_points,
                    grid_reso=query_grid_reso,
                    n_cams=n_cams,
                ),
                ff_cfg=dict(
                    type='MLP',
                    in_channels=d_fusion,
                    hidden_channels=d_fusion*2,
                    out_channels=d_fusion,
                    norm_cfg=None,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                norm_cfg=dict(
                    type='LayerNorm',
                    normalized_shape=d_fusion,
                ),
            ),
        ),
        query_info=dict(
            grid_range=point_range,
            grid_reso=query_grid_reso,
            grid_size=query_grid_size,
            d_model=d_fusion,
        ),
        key_info=dict(
            d_model=d_fusion,
            spatial_shapes=key_shapes,
        ),
    ),
    backbone2d=dict(
        type='SECOND',
        in_channels=d_fusion,
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
        in_channels=2*256,
        shared_conv_channels=64,
        init_bias=-2.20,
        heads={
            'heatmap': (3, 2),
            'offset': (2, 2),
            'height': (1, 2),
            'size': (3, 2),
            'heading': (2, 2)
        },  # (output_channel, num_conv)
    ),
)
model = FI.create(model_config).half().cuda()

points = [torch.rand(1000, point_dim, dtype=torch.float32).cuda(),
          torch.rand(1200, point_dim, dtype=torch.float32).cuda()]
images = torch.rand(2, n_cams, 3, in_h, in_w, dtype=torch.float32).cuda()
calibs = torch.rand(2, n_cams, 3, 4, dtype=torch.float32).cuda()

batch = dict(input=dict(points=points, images=images, calibs=calibs))

with autocast():
    out = model(batch)
    print(out)
