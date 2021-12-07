from mdet.utils.factory import FI
import mdet.data
import mdet.model
import torch

import open3d as o3d
from mdet.ops.voxelization import Voxelize

voxel_size = [0.32,0.32,10]
point_range= [-40,-40,-5,40,40,5]

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

dataset = FI.create(dataset_config)
voxelization = FI.create(voxelization_config)
pillar_feature_net = FI.create(pillar_feature_net_config)


voxelization_result = []
voxelization_result.append(voxelization(torch.from_numpy(dataset[299]['pcd']).cpu()))
voxelization_result.append(voxelization(torch.from_numpy(dataset[499]['pcd']).cpu()))

persudo_image = pillar_feature_net(voxelization_result)
print(persudo_image.shape)

