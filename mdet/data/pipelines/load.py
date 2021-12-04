from mdet.utils.factory import FI
import mdet.utils.io as io
import numpy as np
import mdet.utils.rigid as rigid


@FI.register
class WaymoLoadSweep(object):
    def __init__(self, load_dim=5, load_nsweep=1, **kwargs):
        self.load_nsweep = load_nsweep
        self.load_dim = load_dim

    def __call__(self, sample, info):
        sweep_info_list = info['sweeps']
        assert len(sweep_info_list) >= self.load_nsweep

        sweeps = []
        for sweep_info in sweep_info_list[:self.load_nsweep]:
            sweeps.append({
                'timestamp': sweep_info['timestamp'],
                'tf_map_vehicle': sweep_info['tf_map_vehicle'],
                'pcd': io.load(sweep_info['pcd_path'])[:, :self.load_dim],
            })

        sample.update({'sweeps': sweeps})


@FI.register
class WaymoLoadAnno(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, sample, info):
        anno = io.load(info['anno_path'])

        boxes = np.stack([object['box'] for object in anno['objects']], axis=0)
        types = np.array([object['type']
                         for object in anno['objects']], dtype=np.int32)
        num_points = np.array([object['num_points']
                              for object in anno['objects']], dtype=np.int32)

        sample.update(
            {'gt': {'boxes': boxes, 'types': types, 'num_points': num_points}})


@FI.register
class MergeSweep(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, sample, info):
        pcds = [sample['sweeps'][0]['pcd']]
        tf_map_vehicle0 = sample['sweeps'][0]['tf_map_vehicle']
        for i in range(1, len(sample['sweeps'])):
            pcd = sample['sweeps'][i]['pcd']
            tf_map_vehicle = sample['sweeps'][i]['tf_map_vehicle']
            tf = rigid.between(tf_map_vehicle0, tf_map_vehicle)
            pcds.append(rigid.transform(tf, pcd))

        sample.update({'pcd': np.concatenate(pcds, axis=0)})
        sample.pop('sweeps')
