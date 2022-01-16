import numpy as np
from mdet.utils.factory import FI
from mdet.data.datasets.base_dataset import MDet3dDataset
import mdet.utils.io as io
import mdet.utils.rigid as rigid
from mdet.core.annotation import Annotation3d
from mdet.core.pointcloud import Pointcloud
import os


@FI.register
class WaymoDet3dDataset(MDet3dDataset):
    def __init__(self,
                 info_path,
                 load_opt={},
                 transforms=[],
                 codec=None,
                 filter=None):
        super().__init__(info_path, transforms, codec, filter)
        self.load_opt = load_opt

        self.type_raw_to_task = {}
        self.type_id_to_name = []
        for task_specific_type, (type_name, raw_type_list) in enumerate(
                load_opt['types']):
            self.type_id_to_name.append(type_name)
            for raw_type in raw_type_list:
                self.type_raw_to_task[raw_type] = task_specific_type

        self.pcd_loader = WaymoNSweepLoader(
            load_opt['load_dim'], load_opt['nsweep'])

    def load_meta(self, sample, info):
        anno = io.load(info['anno_path'])
        frame_id = anno['frame_id']
        seq_name = anno['seq_name']
        sample_name = f'{seq_name}-{frame_id}'

        # update sample's meta
        sample['meta'] = dict(sample_name=sample_name)

    def load_data(self, sample, info):
        # load pcd
        pcd = self.pcd_loader(info['sweeps'])

        # update sample's data
        sample['data'] = dict(pcd=pcd)

    def load_anno(self, sample, info):
        anno = io.load(info['anno_path'])

        boxes = np.empty((0, 8), dtype=np.float32)
        types = np.empty((0, ), dtype=np.int32)
        num_points = np.empty((0, ), dtype=np.int32)
        box_list, type_list, num_points_list = [], [], []
        for object in anno['objects']:
            raw_type = object['type']
            if raw_type in self.type_raw_to_task:
                raw_box = object['box']
                box_center = raw_box[:3]
                box_extend = raw_box[3:6] / 2
                box_rotation = np.concatenate(
                    [np.cos(raw_box[[6]]),
                     np.sin(raw_box[[6]])])
                box = np.concatenate([box_center, box_extend, box_rotation])
                box_list.append(box)
                type_list.append(self.type_raw_to_task[raw_type])
                num_points_list.append(object['num_points'])
        if len(box_list) > 0:
            boxes = np.stack(box_list, axis=0)
            types = np.array(type_list, dtype=np.int32)
            num_points = np.array(num_points_list, dtype=np.int32)

        # update sample's anno and meta
        sample['anno'] = Annotation3d(boxes=boxes,
                                      types=types,
                                      num_points=num_points)
        sample['meta']['type_name'] = self.type_id_to_name

    def format(self, output, pred_path=None, gt_path=None):
        r'''
        format output to dataset specific format for submission and evaluation.
        if output_path is None, a tmp file will be used
        '''
        return pred_path, gt_path

    def evaluate(self, pred_path, gt_path):
        r'''
        evaluate metrics
        '''
        return {'accuracy': 0.0}


@FI.register
class WaymoNSweepLoader(object):
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
            tf_map_vehicle = sweep_info['tf_map_vehicle']
            if i == 0:
                tf_map_vehicle0 = tf_map_vehicle

            pcd_path = sweep_info['pcd_path']
            if not pcd_path or not os.path.exists(f'{pcd_path}.gz'):
                print(f'{pcd_path} not exists')
                continue

            pcd = io.load(sweep_info['pcd_path'], compress=True)
            if i > 0:
                tf_cur_past = rigid.between(tf_map_vehicle0, tf_map_vehicle)
                pcd = rigid.transform(tf_cur_past, pcd)
            pcd_list.append(pcd)

        points = np.concatenate(pcd_list, axis=0)

        return Pointcloud(points=points[:, :self.load_dim])
