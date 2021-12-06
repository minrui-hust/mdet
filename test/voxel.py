from mdet.utils.factory import FI
import mdet.data
import torch

import open3d as o3d
from mdet.ops.voxelization import Voxelize

dataset_config = dict(
    type='WaymoDataset',
    info_path='/data/tmp/waymo/training_info.pkl',
    transforms_cfg=[
        dict(type='WaymoLoadSweep', load_nsweep=1),
        dict(type='WaymoLoadAnno'),
        dict(type='MergeSweep'),
    ],
)

dataset = FI.create(dataset_config)

for _ in range(100):
    voxels, coords, point_num, voxel_num = Voxelize(torch.from_numpy(dataset[300]['pcd']).cpu(), [-40,-40,-5,40,40,5], [0.32,0.32,10])

print('cpu done')

for _ in range(100):
    voxels, coords, point_num, voxel_num = Voxelize(torch.from_numpy(dataset[300]['pcd']).cuda(), [-40,-40,-5,40,40,5], [0.32,0.32,10])
print('gpu done')



