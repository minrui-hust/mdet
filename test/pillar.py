from mdet.utils.factory import FI
import mdet.data
import mdet.model
import torch

import open3d as o3d
from mdet.ops.voxelization import Voxelize

voxel_size = [0.32,0.32,10]
point_range= [-64,-64,-5,64,64,5]

dataset_config = dict(
    type='WaymoDataset',
    info_path='/data/tmp/waymo/training_info.pkl',
    transforms_cfg=[
        dict(type='WaymoLoadSweep', load_nsweep=1),
        dict(type='WaymoLoadAnno'),
        dict(type='MergeSweep'),
    ],
)

voxelization_config = dict(type='PillarVoxelization',
                           voxel_size=voxel_size,
                           point_range=point_range,
                           max_points=32,
                           max_voxels=15000)

pillar_feature_net_config = dict(type='PillarFeatureNet',
                                 pillar_feat=[('position', 3), 
                                              ('attribute', 2), 
                                              ('center_offset', 2), 
                                              ('mean_offset', 3), 
                                              ('distance', 1)],
                                 voxel_size=voxel_size,
                                 point_range=point_range,
                                 pfn_channels=(64,))

backbone2d_config = dict(type='SECOND', 
                        in_channels=64,
                        out_channels=[128, 128, 256],
                        layer_nums=[3, 5, 5],
                        layer_strides=[2, 2, 2],
                        )

neck_config = dict(type='SECONDFPN',
                   in_channels=[128, 128, 256],
                   out_channels=[128, 128, 128],
                   upsample_strides=[1, 2, 4])

head_config = dict(type='CenterHead',
                   in_channels = 3*128,
                   shared_conv_channels=64,
                   heads={'heatmap':(3, 2), 
                          'offset': (2, 2), 
                          'height': (1, 2), 
                          'size':(3, 2), 
                          'rot':(2, 2)}, # (output_channel, num_conv)
                   init_bias=-2.20
                  )

dataset = FI.create(dataset_config)
voxelization = FI.create(voxelization_config).cuda()
pillar_feature_net = FI.create(pillar_feature_net_config).cuda()
backbone2d = FI.create(backbone2d_config).cuda()
neck = FI.create(neck_config).cuda()
head = FI.create(head_config).cuda()


voxelization_result = []
voxelization_result.append(voxelization(torch.from_numpy(dataset[299]['pcd']).cuda()))
voxelization_result.append(voxelization(torch.from_numpy(dataset[499]['pcd']).cuda()))
persudo_image = pillar_feature_net(voxelization_result)
layered_feature = backbone2d(persudo_image)
feat = neck(layered_feature)
out_dict = head(feat)
for name, value in out_dict.items():
    print(f'{name}: {value.shape}')

