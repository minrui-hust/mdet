from functools import partial
import multiprocessing as mp
import os
import json

from mai.utils import FI
from mai.utils import io
import numpy as np
from tqdm import tqdm

from mdet.core.box_np_ops import points_in_boxes
import mdet.utils.rigid as rigid


@FI.register
class ShieldDet3dSummary(object):
    def __init__(self):
        super().__init__()

    def __call__(self, root_path, split, nsweep=1):
        summary(root_path, split, nsweep)


def summary(root_path, split, nsweep=1):
    r'''
    '''
    split_path = os.path.join(root_path, split)
    seq_name_list = list(os.listdir(split_path))

    infos = []
    for seq_name in tqdm(seq_name_list):
        seq_path = os.path.join(split_path, seq_name)
        seq_label_path = os.path.join(seq_path, 'label_3d')
        seq_vehicle_state_path = os.path.join(seq_path, 'vehicle_state')

        seq_extrincs = io.load(os.path.join(seq_path, 'extrinsics.json'))

        frame_name_list = [os.path.splitext(
            fname)[0] for fname in os.listdir(seq_label_path)]
        frame_name_list.sort(key=lambda x: int(x))
        for frame_name in frame_name_list:
            frame_id = int(frame_name)
            frame_anno_path = os.path.join(seq_label_path, f'{frame_name}.txt')
            frame_anno_tf = rigid.from_coeffs(
                np.array(seq_extrincs['label_3d'], dtype=np.float32))
            anno = dict(path=frame_anno_path, tf=frame_anno_tf)

            sweeps = []
            for frame_id in range(frame_id, frame_id - nsweep, -1):
                frame_id = max(frame_id, 0)
                vehicle_state = load_vechile_state(os.path.join(
                    seq_vehicle_state_path, f'{str(frame_id).zfill(6)}.json'))
                scans = []
                for lidar_name in ['lidar_left', 'lidar_right']:
                    pcd_path = os.path.join(
                        seq_path, lidar_name, f'{str(frame_id).zfill(6)}.pcd')
                    tf_vehicle_lidar = rigid.from_coeffs(
                        np.array(seq_extrincs[lidar_name], dtype=np.float32))
                    scan = dict(name=lidar_name,
                                pcd_path=pcd_path,
                                transform=tf_vehicle_lidar)
                    scans.append(scan)
                sweep = dict(vehicle_state=vehicle_state,
                             scans=scans)
                sweeps.append(sweep)

            info = dict(anno=anno,
                        sweeps=sweeps,
                        seq_name=seq_name,
                        frame_name=frame_name)
            infos.append(info)

    io.dump(infos, os.path.join(root_path, f'{split}_info.pkl'), format='pkl')


def load_vechile_state(path):
    r'''
    vehicle state contains:
        timestamp: time stamp
        transform: tf_map_vehicle
        velocity: vel_map_vehicle expressed in vehicle
    '''
    raw_state = io.load(path, 'json')

    #  timestamp = raw_state['timestamp']
    timestamp = 0
    position = np.array([raw_state['position']['x'],
                         raw_state['position']['y'],
                         raw_state['position']['z']], dtype=np.float32)
    orientation = np.array([raw_state['orientation']['w'],
                            raw_state['orientation']['x'],
                            raw_state['orientation']['y'],
                            raw_state['orientation']['z']], dtype=np.float32)
    linear_vel = np.array([raw_state['linear_velocity']['x'],
                           raw_state['linear_velocity']['y'],
                           raw_state['linear_velocity']['z']], dtype=np.float32)
    angular_vel = np.array([raw_state['angular_velocity']['x'],
                            raw_state['angular_velocity']['y'],
                            raw_state['angular_velocity']['z']], dtype=np.float32)

    transform = rigid.from_coeffs(np.concatenate([position, orientation]))
    velocity = np.concatenate([linear_vel, angular_vel])

    return dict(timestamp=timestamp,
                transform=transform,
                velocity=velocity)
