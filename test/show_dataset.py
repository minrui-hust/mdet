from mdet.utils.factory import FI
import mdet.data

import open3d as o3d


types = [('Pedestrian', [2]), ('Cyclist', [4]), ('Vehicle', [1])]

point_range = [-64, -64, -5, 64, 64, 5]
voxel_size = [0.32, 0.32, 10]
voxel_reso = [400, 400, 1]

out_grid_size = [0.64, 0.64]
out_grid_reso = [200, 200]


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
    ],
)
dataset = FI.create(dataset_config)


for data in dataset[250:300]:
    dataset.plot(data)
