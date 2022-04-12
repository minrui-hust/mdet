import math
import math
import os

from mai.utils import FI
from mai.utils import io
import numpy as np
from tqdm import tqdm

from mdet.core.annotation3d import Annotation3d
from mdet.core.pointcloud import Pointcloud
from mdet.data.datasets.mdet3d_dataset import MDet3dDataset
import mdet.utils.rigid as rigid

import pandas as pd


@FI.register
class ShieldDet3dDataset(MDet3dDataset):
    def __init__(self,
                 info_path,
                 load_opt={},
                 filters=[],
                 transforms=[],
                 codec=None,
                 ):
        super().__init__(info_path, filters, transforms, codec)
        self.load_opt = load_opt

        self.pcd_loader = ShieldNSweepLoader(
            load_opt['load_dim'], load_opt['nsweep'])

        self.load_types = load_opt['interest_types']

        self.type_map = dict(Vehicle=0, Cyclist=1, Pedestrian=2)

    def load_meta(self, sample, info):
        seq_name = info['seq_name']
        frame_name = info['frame_name']

        # update sample's meta
        sample['meta'] = dict(seq_name=seq_name,
                              frame_name=frame_name)

    def load_data(self, sample, info):
        # load pcd
        pcd = self.pcd_loader(info['sweeps'])

        # update sample's data
        sample['data'] = dict(pcd=pcd)

    def load_anno(self, sample, info):
        anno_info = info['anno']
        anno_path = anno_info['path']
        anno_tf = anno_info['tf']

        # load label
        label = self.load_label(anno_path)
        boxes = label['boxes']
        types = label['types']

        # transform label into vehicle frame
        # transform box center
        boxes[:, :3] = rigid.transform(anno_tf, boxes[:, :3])

        # transform box rotation
        dir_vect = np.concatenate(
            [boxes[:, 6:], np.zeros((len(boxes), 1))], axis=-1)
        boxes[:, 6:] = (anno_tf[:3, :3]@(dir_vect.T)).T[:, :2]

        # update sample's anno and meta
        sample['anno'] = Annotation3d(boxes=boxes,
                                      types=types)

    def load_label(self, label_path):
        label = pd.read_csv(label_path, delim_whitespace=True,
                            usecols=[1, 7, 8, 9, 10, 11, 12, 13],
                            names=['type', 'l', 'w', 'h', 'x', 'y', 'z', 'r'])

        boxes = label[['x', 'y', 'z', 'l', 'w', 'h', 'r', 'r']].to_numpy()
        boxes[:, 3:6] /= 2
        boxes[:, 6] = np.cos(boxes[:, 6])
        boxes[:, 7] = np.sin(boxes[:, 7])

        types = label['type'].tolist()
        types = np.array([self.type_map[type]
                         for type in types], dtype=np.int32)

        return dict(boxes=boxes.astype(np.float32), types=types.astype(np.int32))

    def format(self, sample_list, pred_path=None, gt_path=None):
        return pred_path, gt_path

    def evaluate(self, pred_path, gt_path):
        return None


@FI.register
class ShieldNSweepLoader(object):
    def __init__(self, load_dim, nsweep=1):
        super().__init__()
        self.load_dim = load_dim
        self.nsweep = nsweep

    def __call__(self, sweep_info_list):
        r'''
        Args:
            sweep_info_list: sweep infos from current to past
        '''
        assert(self.nsweep > 0)
        assert(len(sweep_info_list) >= self.nsweep)

        pcd_list = []
        tf_map_vehicle0 = None
        for i, sweep_info in enumerate(sweep_info_list[:self.nsweep]):
            tf_map_vehicle = sweep_info['vehicle_state']['transform']
            if i == 0:
                tf_map_vehicle0 = tf_map_vehicle

            scan_pcd_list = []
            for scan_info in sweep_info['scans']:
                pcd_path = scan_info['pcd_path']
                tf_vehicle_lidar = scan_info['transform']
                if not pcd_path or not os.path.exists(pcd_path):
                    print(f'{pcd_path} not exists')
                    continue
                # load scan and convert it into vehicle frame
                scan_pcd = np.fromfile(
                    pcd_path, dtype=np.float64).reshape(-1, 6)[:, :self.load_dim]
                scan_pcd = rigid.transform(tf_vehicle_lidar, scan_pcd)
                scan_pcd_list.append(scan_pcd)
            pcd = np.concatenate(scan_pcd_list, axis=0)

            # convert the past pcd into current vehicle frame
            if i > 0:
                tf_cur_past = rigid.between(tf_map_vehicle0, tf_map_vehicle)
                pcd = rigid.transform(tf_cur_past, pcd)
            pcd_list.append(pcd)

        return Pointcloud(points=np.concatenate(pcd_list, axis=0).astype(np.float32))


@FI.register
class ShieldObjectNSweepLoader(object):
    def __init__(self, load_dim, nsweep=1):
        super().__init__()
        self.load_dim = load_dim
        self.nsweep = nsweep

    def __call__(self, sweeps):
        r'''
        Args:
            sweep_info_list: sweep infos from current to past
        '''
        assert(self.nsweep > 0)

        prefix, seq_name, frame_name, object_id = sweeps['prefix'], sweeps[
            'seq_name'], sweeps['frame_name'], sweeps['object_id']

        sweep_data_path = os.path.join(
            prefix, seq_name, f'{frame_name}-{object_id}.pkl')
        points = io.load(sweep_data_path, compress=True)

        return Pointcloud(points=points[:, :self.load_dim])
