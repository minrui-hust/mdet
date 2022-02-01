from mdet.utils.factory import FI
import mdet.data

import open3d as o3d
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader

labels = [('Pedestrian', [2]), ('Cyclist', [4]), ('Vehicle', [1])]

point_range = [-64, -64, -5, 64, 64, 5]
voxel_size = [0.32, 0.32, 10]
voxel_reso = [400, 400, 1]

out_grid_size = [0.64, 0.64]
out_grid_reso = [200, 200]

db_sampler = dict(
    type='GroundTruthSampler',
    info_path='/data/tmp/waymo/training_info_gt.pkl',
    sample_groups=[('Vehicle', 15), ('Pedestrian', 10), ('Cyclist', 10)],
    labels=labels,
    pcd_loader=dict(type='WaymoObjectNSweepLoader', load_dim=5, nsweep=1),
    filters=[dict(type='FilterByNumpoints', min_num_points=5)],
)

dataset = dict(
    type='WaymoDet3dDataset',
    info_path='/data/tmp/waymo/training_info.pkl',
    load_opt=dict(load_dim=5, nsweep=1, labels=labels,),
    transforms=[
        dict(type='PcdIntensityNormlizer'),
        dict(type='PcdObjectSampler', db_sampler=db_sampler),
        #  dict(type='PcdMirrorFlip', mirror_prob=0.5, flip_prob=0.5),
        #  dict(type='PcdGlobalTransform',
        #       rot_range=[-0.78539816, 0.78539816],
        #       scale_range=[0.95, 1.05],
        #       translation_std=[0.5, 0.5, 0]),
        dict(type='PcdRangeFilter', point_range=point_range),
        dict(type='PcdShuffler'),
    ],
    filter=dict(type='IntervalDownsampler', interval=5),
)

dataset = FI.create(dataset)
for i in range(len(dataset)):
    dataset.plot(dataset[i])

dataloader = DataLoader(dataset, batch_size=2,
                        num_workers=4, collate_fn=lambda x: x)

#  dataset.plot(dataset[0])
for batch in tqdm(dataloader):
    pass

#  for data in dataset[250:300]:
#      dataset.plot(data)
