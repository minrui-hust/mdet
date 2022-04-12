import os

from mai.utils import FI
from mai.utils import io

from tqdm import tqdm

from .dataset import ShieldDet3dDataset

from mdet.core.box_np_ops import points_in_boxes


@FI.register
class ShieldDet3dGtDatabaseCreator(object):
    def __init__(self):
        super().__init__()

    def __call__(self, root_path, split):
        create_gt_database(root_path, split)


def create_gt_database(root_path, split):
    r'''
    create ground truth database
    '''
    # first create the dataset to tranverse
    dataset = ShieldDet3dDataset(
        info_path=os.path.join(root_path, f'{split}_info.pkl'),
        load_opt=dict(load_dim=4, nsweep=1, interest_types=[]),
    )

    object_info_dict = {0: [], 1: [], 2: []}
    gt_database_folder = os.path.join(root_path, 'gt_database', split)
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        create_gt_database_one_sample(
            data, gt_database_folder, object_info_dict)

    for type, info_list in object_info_dict.items():
        print(f'type {type}: {len(info_list)}')

    print('wait for writing data info...')
    io.dump(object_info_dict, os.path.join(root_path, f'{split}_info_gt.pkl'))


def create_gt_database_one_sample(sample, gt_database_folder, object_info_dict):
    seq_name = sample['meta']['seq_name']
    frame_name = sample['meta']['frame_name']
    seq_folder = os.path.join(gt_database_folder, seq_name)
    os.makedirs(seq_folder, exist_ok=True)

    boxes = sample['anno'].boxes
    types = sample['anno'].types
    points = sample['data']['pcd'].points

    point_indices = points_in_boxes(points, boxes)
    for i in range(len(boxes)):
        object_pcd = points[point_indices[:, i]]
        object_fname = f'{frame_name}-{i}.pkl'
        object_path = os.path.join(seq_folder, object_fname)

        # save object pcd
        io.dump(object_pcd, object_path, compress=True)

        # update info
        type = types[i]
        box = boxes[i]
        num_points = object_pcd.shape[0]
        sweeps = dict(prefix=gt_database_folder, seq_name=seq_name,
                      frame_name=frame_name, object_id=i)
        object_info_dict[type].append(
            dict(box=box, type=type, num_points=num_points, sweeps=sweeps))
