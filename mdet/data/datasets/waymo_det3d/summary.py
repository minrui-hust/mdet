from functools import partial
import multiprocessing as mp
import os

from mai.utils import FI
from mai.utils import io
import numpy as np
from tqdm import tqdm

from mdet.core.box_np_ops import points_in_boxes
import mdet.utils.rigid as rigid


@FI.register
class WaymoDet3dSummary(object):
    def __init__(self):
        super().__init__()

    def __call__(self, root_path, split, nsweep=5):
        summary(root_path, split, nsweep)


@FI.register
class WaymoDet3dGtDatabaseCreator(object):
    def __init__(self):
        super().__init__()

    def __call__(self, root_path, split, summary_only=False, nsweep=2):
        if not summary_only:
            create_gt_database(root_path, split)
        summary_gt_database(root_path, split, nsweep)


def summary(root_path, split, nsweep=5):
    anno_path = os.path.join(root_path, split, 'annos')
    frame_name_list = sort_frame(list(os.listdir(anno_path)))

    infos = []
    for cur_frame_name in tqdm(frame_name_list):
        cur_anno_path = os.path.join(root_path, split, 'annos', cur_frame_name)

        cur_anno = io.load(cur_anno_path, format='pkl')
        cur_frame_id = cur_anno['frame_id']
        cur_seq_id = cur_anno['seq_name']

        sweeps = []
        for frame_id in range(cur_frame_id, cur_frame_id - nsweep, -1):
            frame_id = max(frame_id, 0)
            anno_path = os.path.join(root_path, split, 'annos',
                                     f'{cur_seq_id}-{frame_id}.pkl')
            pcd_path = os.path.join(root_path, split, 'pcds',
                                    f'{cur_seq_id}-{frame_id}.pkl')
            anno = io.load(anno_path, format='pkl')

            sweep = {
                'timestamp': anno['timestamp'],
                'tf_map_vehicle': anno['tf_map_vehicle'],
                'pcd_path': pcd_path,
            }
            sweeps.append(sweep)

        info = {
            'anno_path':
            os.path.join(root_path, split, 'annos',
                         f'{cur_seq_id}-{cur_frame_id}.pkl'),
            'sweeps':
            sweeps,
        }
        infos.append(info)

    io.dump(infos, os.path.join(root_path, f'{split}_info.pkl'), format='pkl')


def create_gt_database(root_path, split):
    r'''
    create ground truth database
    '''

    anno_folder = os.path.join(root_path, split, 'annos')
    frame_name_list = sort_frame(list(os.listdir(anno_folder)))

    seq_frame_list_dict = {}
    for frame_name in frame_name_list:
        anno_path = os.path.join(anno_folder, frame_name)
        anno = io.load(anno_path)
        seq_id = anno['seq_name']
        if seq_id not in seq_frame_list_dict:
            seq_frame_list = [frame_name]
            seq_frame_list_dict[seq_id] = seq_frame_list
        else:
            seq_frame_list_dict[seq_id].append(frame_name)

    print(f'Total {len(seq_frame_list_dict)} sequence to process')

    with mp.Pool(int(mp.cpu_count() / 2)) as p:
        p.map(
            partial(create_gt_database_one_seq,
                    root_path=root_path,
                    split=split), seq_frame_list_dict.items())


def create_gt_database_one_seq(seq_item, root_path, split):
    gt_database_folder = os.path.join(root_path, 'gt_database', split)

    # make sequence folder
    seq_id, seq_frame_list = seq_item
    seq_folder = os.path.join(gt_database_folder, seq_id)
    os.makedirs(seq_folder, exist_ok=True)

    anno_folder = os.path.join(root_path, split, 'annos')
    pcd_folder = os.path.join(root_path, split, 'pcds')

    for frame_name in tqdm(seq_frame_list):
        anno_path = os.path.join(anno_folder, frame_name)
        pcd_path = os.path.join(pcd_folder, frame_name)

        anno = io.load(anno_path)
        seq_id = anno['seq_name']
        frame_id = anno['frame_id']
        tf_map_vehicle = anno['tf_map_vehicle']
        objects = anno['objects']
        if len(objects) <= 0:
            continue

        objects_box_list = [object['box'] for object in objects]
        objects_box = normlize_boxes(np.stack(objects_box_list, axis=0))

        pcd = io.load(pcd_path, compress=True)
        point_indices = points_in_boxes(pcd, objects_box)

        for object_id in range(objects_box.shape[0]):
            object_pcd = pcd[point_indices[:, object_id]]
            object_data = (object_pcd, tf_map_vehicle)

            object_name = objects[object_id]['name']
            object_data_filename = f'{frame_id}-{object_name}.pkl'
            pcd_path = os.path.join(seq_folder, object_data_filename)

            io.dump(object_data, pcd_path, compress=True)


def summary_gt_database(root_path, split, nsweep=2):
    r'''
    summary the gt database
    '''
    gt_database_folder = os.path.join(root_path, 'gt_database', split)

    anno_folder = os.path.join(root_path, split, 'annos')
    frame_name_list = sort_frame(list(os.listdir(anno_folder)))

    # UNKNOWN = 0, VEHICLE = 1, PEDESTRIAN = 2, SIGN = 3, CYCLIST = 4,
    object_info_dict = {0: [], 1: [], 2: [], 3: [], 4: []}

    for frame_name in tqdm(frame_name_list):
        anno = io.load(os.path.join(anno_folder, frame_name))
        seq_id = anno['seq_name']
        frame_id = anno['frame_id']
        objects = anno['objects']
        if len(objects) <= 0:
            continue

        objects_box_list = [object['box'] for object in objects]
        objects_box = normlize_boxes(np.stack(objects_box_list, axis=0))

        for i, object in enumerate(objects):
            type = object['type']
            object_id = object['name']
            num_points = object['num_points']
            sweeps = dict(prefix=gt_database_folder, seq_id=seq_id,
                          frame_id=frame_id, object_id=object_id)

            object_info_dict[type].append(dict(box=objects_box[i],
                                               type=type,
                                               num_points=num_points,
                                               sweeps=sweeps,
                                               ))

    for type, info_list in object_info_dict.items():
        print(f'type {type}: {len(info_list)}')

    print('wait for writing data info...')
    io.dump(object_info_dict, os.path.join(root_path, f'{split}_info_gt.pkl'))


def sort_frame(frame_name_list):
    seq_frame_list = []
    for frame_name in frame_name_list:
        frame_name = os.path.splitext(frame_name)[0]
        seq, frame = frame_name.split('-')
        seq_frame_list.append((seq, int(frame)))
    seq_frame_list.sort()

    return [f'{p[0]}-{p[1]}.pkl' for p in seq_frame_list]


def normlize_boxes(boxes):
    r'''
    convert waymo raw box to standard box
    '''
    center = boxes[:, :3]
    extend = boxes[:, 3:6] / 2
    rotation = np.stack([np.cos(boxes[:, 6]), np.sin(boxes[:, 6])], axis=-1)
    return np.concatenate([center, extend, rotation], axis=-1)
