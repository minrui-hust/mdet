import numpy as np
from mdet.utils.factory import FI
from mdet.data.datasets.base_dataset import MDet3dDataset
import mdet.utils.io as io
import mdet.utils.rigid as rigid
from mdet.core.annotation import Annotation3d
from mdet.core.pointcloud import Pointcloud


@FI.register
class WaymoDet3dDataset(MDet3dDataset):
    def __init__(self, info_path, load_opt={}, transforms=[], codec=None, filter=None):
        super().__init__(info_path, transforms, codec, filter)
        self.load_opt = load_opt

        self.type_raw_to_task = {}
        self.type_id_to_name = []
        for task_specific_type, (type_name, raw_type_list) in enumerate(load_opt['types']):
            self.type_id_to_name.append(type_name)
            for raw_type in raw_type_list:
                self.type_raw_to_task[raw_type] = task_specific_type

    def load_meta(self, sample, info):
        anno = io.load(info['anno_path'])
        frame_id = anno['frame_id']
        seq_name = anno['seq_name']
        sample_name = f'{seq_name}-{frame_id}'

        # update sample's meta
        sample['meta'] = dict(sample_name=sample_name)

    def load_data(self, sample, info):
        sweep_info_list = info['sweeps']
        num_sweeps = self.load_opt['num_sweeps']
        load_dim = self.load_opt['load_dim']
        assert len(sweep_info_list) >= num_sweeps

        # load sweeps
        sweeps = []
        for sweep_info in sweep_info_list[:num_sweeps]:
            sweeps.append({
                'timestamp': sweep_info['timestamp'],
                'tf_map_vehicle': sweep_info['tf_map_vehicle'],
                'pcd': io.load(sweep_info['pcd_path'])[:, :load_dim],
            })

        # merge sweeps
        pcds = [sweeps[0]['pcd']]
        tf_map_vehicle0 = sweeps[0]['tf_map_vehicle']
        for i in range(1, len(sweeps)):
            pcd = sweeps[i]['pcd']
            tf_map_vehicle = sweeps[i]['tf_map_vehicle']
            tf = rigid.between(tf_map_vehicle0, tf_map_vehicle)
            pcds.append(rigid.transform(tf, pcd))

        # update sample's data
        sample['data'] = dict(pcd=Pointcloud(
            points=np.concatenate(pcds, axis=0)))

    def load_anno(self, sample, info):
        anno = io.load(info['anno_path'])

        boxes = np.empty((0, 7), dtype=np.float32)
        types = np.empty((0,), dtype=np.int32)
        num_points = np.empty((0,), dtype=np.int32)
        box_list, type_list, num_points_list = [], [], []
        for object in anno['objects']:
            raw_type = object['type']
            if raw_type in self.type_raw_to_task:
                raw_box = object['box']
                box_center = raw_box[:3]
                box_extend = raw_box[3:6] / 2
                box_rotation = np.concatenate([np.cos(raw_box[[6]]), np.sin(raw_box[[6]])])
                box = np.concatenate( [box_center, box_extend, box_rotation] )
                box_list.append(box)
                type_list.append(self.type_raw_to_task[raw_type])
                num_points_list.append(object['num_points'])
        if len(box_list) > 0:
            boxes = np.stack(box_list, axis=0)
            types = np.array(type_list, dtype=np.int32)
            num_points = np.array(num_points_list, dtype=np.int32)

        # update sample's anno and meta
        sample['anno'] = Annotation3d(
            boxes=boxes, types=types, num_points=num_points)
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
