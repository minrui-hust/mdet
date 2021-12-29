from mdet.utils.factory import FI
import mdet.data
import torch

from mdet.ops.voxelization import Voxelize

types = [('Vehicle', [1])]

point_range = [-75, -75, -2, 75, 75, 4.0]
voxel_size = [0.1, 0.1, 0.15]
voxel_reso = [1500, 1500, 40]

dataset_cfg = dict(
    type='WaymoDet3dDataset',
    info_path='/data/tmp/waymo/training_info.pkl',
    load_opt=dict(
        load_dim=5,
        num_sweeps=1,
        types=types,
    ),
    transforms=[
        dict(type='RangeFilter', point_range=point_range),
    ],
)


dataset = FI.create(dataset_cfg)

voxels, coords, point_num, voxel_num = Voxelize(torch.from_numpy(
    dataset[0]['data']['pcd'].points).cpu(), point_range, voxel_size, voxel_reso, 5, 150000, 'nearest')
print(voxels.shape)
print(voxel_num)

voxels, coords, point_num, voxel_num = Voxelize(torch.from_numpy(
    dataset[0]['data']['pcd'].points).cuda(), point_range, voxel_size, voxel_reso, 0, 150000, 'mean')
print(voxels.shape)
print(voxel_num)
