from mdet.utils.factory import FI
import mdet.data

import open3d as o3d

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

for data in dataset:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.Vector3dVector(data['pcd'][:, :3])
    o3d.draw_geometries([pcd])
