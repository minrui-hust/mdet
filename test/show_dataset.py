from mdet.utils.factory import FI
import mdet.data
import numpy as np

from enum import IntEnum

import open3d as o3d
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader


class RawType:
    Vehicle = 1
    Cyclist = 4
    Pedestrian = 2


class NewType:
    Vehicle0 = 0
    Vehicle1 = 1
    Vehicle2 = 2
    Vehicle3 = 3
    Cyclist = 4
    Pedestrian = 5


class Label:
    Vehicle = 0
    Cyclist = 1
    Pedestrian = 2


labels = {
    Label.Vehicle: [NewType.Vehicle0, NewType.Vehicle1, NewType.Vehicle2, NewType.Vehicle3],
    Label.Cyclist: [NewType.Cyclist],
    Label.Pedestrian: [NewType.Pedestrian],
}


point_range = [-64, -64, -5, 64, 64, 5]
voxel_size = [0.32, 0.32, 10]
voxel_reso = [400, 400, 1]

out_grid_size = [0.64, 0.64]
out_grid_reso = [200, 200]

margin = 2.0

retype = dict(
    type='WaymoCarRetyper',
    raw_car_type=RawType.Vehicle,
    new_car_types={
        NewType.Vehicle0: (0, 3.5),
        NewType.Vehicle1: (3.5, 8),
        NewType.Vehicle2: (8, 15),
        NewType.Vehicle3: (15, 30),
    },
    other_types={
        RawType.Cyclist: NewType.Cyclist,
        RawType.Pedestrian: NewType.Pedestrian
    },
)

db_sampler = dict(
    type='GroundTruthSamplerV2',
    info_path='/data/tmp/waymo/training_info_gt.pkl',
    #  info_path='/data/waymo/det3d/training_info_gt.pkl',
    pcd_loader=dict(type='WaymoObjectNSweepLoader', load_dim=5, nsweep=1),
    interest_types=[RawType.Vehicle, RawType.Cyclist, RawType.Pedestrian],
    retype=retype,
    filters=[
        dict(
            type='FilterByNumpointsV2',
            min_points_groups={
                NewType.Vehicle0: 100,
                NewType.Vehicle1: 100,
                NewType.Vehicle2: 100,
                NewType.Vehicle3: 100,
                NewType.Cyclist: 70,
                NewType.Pedestrian: 50,
            }
        ),
    ],
)

dataset = dict(
    type='WaymoDet3dDataset',
    info_path='/data/tmp/waymo/training_info.pkl',
    #  info_path='/data/waymo/det3d/training_info.pkl',
    load_opt=dict(load_dim=5, nsweep=1, interest_types=[
                  RawType.Vehicle, RawType.Cyclist, RawType.Pedestrian]),
    transforms=[
        dict(type='AnnoRetyper', retype=retype),
        dict(type='PointNumFilterV2', groups={
            NewType.Vehicle0: 100,
            NewType.Vehicle1: 100,
            NewType.Vehicle2: 100,
            NewType.Vehicle3: 100,
            NewType.Cyclist: 70,
            NewType.Pedestrian: 50,
        }),
        dict(type='PcdObjectSamplerV2',
             db_sampler=db_sampler, sample_groups={
                 NewType.Vehicle0: 3,
                 NewType.Vehicle1: 10,
                 NewType.Vehicle2: 5,
                 NewType.Vehicle3: 5,
                 NewType.Cyclist: 8,
                 NewType.Pedestrian: 8,
             }),
        #  dict(type='PcdLocalTransform',
        #       rot_range=[-0.17, 0.17], translation_std=[0.5, 0.5, 0], num_try=50),
        dict(type='PcdMirrorFlip', mirror_prob=0.5, flip_prob=0.5),
        dict(type='PcdGlobalTransform',
             rot_range=[-0.78539816, 0.78539816],
             scale_range=[0.95, 1.05],
             translation_std=[0.5, 0.5, 0]),
        dict(type='PcdRangeFilter', point_range=point_range, margin=margin),
        dict(type='PcdIntensityNormlizer'),
        dict(type='PcdShuffler'),
    ],
    #  filter=dict(type='IntervalDownsampler', interval=20),
)


# codecs config
codec = dict(
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
        ),
    ),
    decode_cfg=dict(
        nms_cfg=dict(
            pre_num=4096,
            post_num=500,
            overlap_thresh=0.7,
        ),
        valid_thresh=0.1,
    ),
    loss_cfg=dict(
        head_weight={
            'heatmap': 1.0,
            'offset': 2 * 2.0,
            'height': 1 * 2.0,
            'size': 3 * 2.0,
            'heading': 2 * 2.0,
        },
        alpha=2.0,
        beta=4.0,
    ),
)


dataset = FI.create(dataset)
codec = FI.create(codec)
dataset.codec = codec

for i in range(len(dataset)):
    dataset.plot(dataset[i])

dataloader = DataLoader(dataset, batch_size=2,
                        num_workers=0, collate_fn=codec.get_collater())

#  dataset.plot(dataset[0])
for batch in tqdm(dataloader):
    pass
    #  print(batch)
