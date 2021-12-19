import numpy as np
from mdet.utils.factory import FI
from mdet.data.datasets.base_dataset import MDet3dDataset
import mdet.utils.io as io
import mdet.utils.rigid as rigid
from mdet.core.annotation import Annotation3d
from mdet.core.pointcloud import Pointcloud


@FI.register
class WaymoDet3dDataset(MDet3dDataset):
    def __init__(self, info_path, load_opt={}, transforms=[], filter=None):
        super().__init__(info_path, transforms, filter)
        self.load_opt = load_opt

        self.type_raw_to_task = {}
        self.type_id_to_name = []
        for task_specific_type, (type_name, raw_type_list) in enumerate(load_opt['types']):
            self.type_id_to_name.append(type_name)
            for raw_type in raw_type_list:
                self.type_raw_to_task[raw_type] = task_specific_type

    def load_pcd(self, sample, info):
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

        # update sample
        sample['pcd'] = Pointcloud(points=np.concatenate(pcds, axis=0))

    def load_anno(self, sample, info):
        anno = io.load(info['anno_path'])

        boxes, types, num_points = [], [], []
        for object in anno['objects']:
            raw_type = object['type']
            if raw_type in self.type_raw_to_task:
                boxes.append(object['box'])
                types.append(self.type_raw_to_task[raw_type])
                num_points.append(object['num_points'])
        boxes = np.stack(boxes, axis=0)
        types = np.array(types, dtype=np.int32)
        num_points = np.array(num_points, dtype=np.int32)

        # update sample
        sample.update({
            'anno': Annotation3d(boxes=boxes, types=types, num_points=num_points),
            'type_name': self.type_id_to_name,
        })

    def format(self, result, output_path):
        raise NotImplementedError

    def evaluate(self, predict_path, gt_path):
        raise NotImplementedError
