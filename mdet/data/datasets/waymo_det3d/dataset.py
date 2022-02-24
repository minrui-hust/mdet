import math
import math
import os

import numpy as np
from tqdm import tqdm
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

from mdet.core.annotation import Annotation3d
from mdet.core.pointcloud import Pointcloud
from mdet.data.datasets.base_dataset import MDet3dDataset
from mdet.utils.factory import FI
import mdet.utils.io as io
import mdet.utils.rigid as rigid


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

        self.pcd_loader = WaymoNSweepLoader(
            load_opt['load_dim'], load_opt['nsweep'])

        self.load_types = load_opt['interest_types']

    def load_meta(self, sample, info):
        anno = io.load(info['anno_path'])
        frame_id = anno['frame_id']
        seq_name = anno['seq_name']
        stamp = anno['timestamp']
        sample_name = f'{seq_name}-{frame_id}'

        # update sample's meta
        sample['meta'] = dict(seq_name=seq_name,
                              frame_id=frame_id,
                              stamp=stamp,
                              name=sample_name)

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
            if object['type'] in self.load_types:
                box_list.append(self._normlize_box(object['box']))
                type_list.append(object['type'])
                num_points_list.append(object['num_points'])
        if len(box_list) > 0:
            boxes = np.stack(box_list, axis=0)
            types = np.array(type_list, dtype=np.int32)
            num_points = np.array(num_points_list, dtype=np.int32)

        # update sample's anno and meta
        sample['anno'] = Annotation3d(boxes=boxes,
                                      types=types,
                                      num_points=num_points)

    def format(self, sample_list, pred_path=None, gt_path=None):
        r'''
        format output to dataset specific format for submission and evaluation.
        if output_path is None, a tmp file will be used
        Args:
            sample_list: sample list to format, must contains 'anno', 'pred', 'meta'
        '''
        if not (pred_path is None and gt_path is None):
            meta_list = [sample['meta'] for sample in sample_list]

        # process prediction
        if pred_path is not None:
            print('Formatting predictions...')
            pred_list = [sample['pred'] for sample in sample_list]
            pred_pb = self._format_anno_list(pred_list, meta_list)
            io.dump(pred_pb, pred_path)
            print(f'Save formatted predictions into {pred_path}')

        # process anno
        if gt_path is not None:
            print('Formatting groundtruth...')
            gt_list = [sample['anno'] for sample in sample_list]
            gt_pb = self._format_anno_list(gt_list, meta_list)
            io.dump(gt_pb, gt_path)
            print(f'Save formatted groundtruth into {gt_path}')

        return pred_path, gt_path

    def evaluate(self, pred_path, gt_path):
        r'''
        evaluate metrics
        '''
        import subprocess
        ret_bytes = subprocess.check_output(
            f'mdet/data/datasets/waymo_det3d/compute_detection_metrics_main {pred_path} {gt_path}', shell=True)
        ret_texts = ret_bytes.decode('utf-8')
        print(ret_texts)

        # parse the text to get ap_dict
        ap_dict = {
            'Vehicle/L1 mAP': 0.0,
            'Vehicle/L1 mAPH': 0.0,
            'Vehicle/L2 mAP': 0.0,
            'Vehicle/L2 mAPH': 0.0,
            'Pedestrian/L1 mAP': 0.0,
            'Pedestrian/L1 mAPH': 0.0,
            'Pedestrian/L2 mAP': 0.0,
            'Pedestrian/L2 mAPH': 0.0,
            'Sign/L1 mAP': 0.0,
            'Sign/L1 mAPH': 0.0,
            'Sign/L2 mAP': 0.0,
            'Sign/L2 mAPH': 0.0,
            'Cyclist/L1 mAP': 0.0,
            'Cyclist/L1 mAPH': 0.0,
            'Cyclist/L2 mAP': 0.0,
            'Cyclist/L2 mAPH': 0.0,
            'Overall/L1 mAP': 0.0,
            'Overall/L1 mAPH': 0.0,
            'Overall/L2 mAP': 0.0,
            'Overall/L2 mAPH': 0.0,
        }
        mAP_splits = ret_texts.split('mAP ')
        mAPH_splits = ret_texts.split('mAPH ')
        for idx, key in enumerate(ap_dict.keys()):
            split_idx = int(idx / 2) + 1
            if idx % 2 == 0:  # mAP
                ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
            else:  # mAPH
                ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
        ap_dict['Overall/L1 mAP'] = (ap_dict['Vehicle/L1 mAP'] +
                                     ap_dict['Pedestrian/L1 mAP'] + ap_dict['Cyclist/L1 mAP']) / 3
        ap_dict['Overall/L1 mAPH'] = (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
                                      ap_dict['Cyclist/L1 mAPH']) / 3
        ap_dict['Overall/L2 mAP'] = (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
                                     ap_dict['Cyclist/L2 mAP']) / 3
        ap_dict['Overall/L2 mAPH'] = (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
                                      ap_dict['Cyclist/L2 mAPH']) / 3

        # do not care about sign
        ap_dict.pop('Sign/L1 mAP')
        ap_dict.pop('Sign/L1 mAPH')
        ap_dict.pop('Sign/L2 mAP')
        ap_dict.pop('Sign/L2 mAPH')

        return ap_dict

    def _format_anno_list(self, anno_list, meta_list):
        r'''
        format Annotation3d into dataset specific format for evaluation and submission
        Args:
            anno: the annotation in Annotation3d format
            meta: sample meta
        Return:
            pb object
        '''
        objects = metrics_pb2.Objects()
        for anno, meta in tqdm(zip(anno_list, meta_list)):
            det_boxes = anno.boxes
            det_types = anno.types
            det_scores = anno.scores
            det_num_points = anno.num_points

            # in case of we have type re mapping
            if 'type_new_to_raw' in meta:
                type_new_to_raw = meta['type_new_to_raw']
                det_types = [type_new_to_raw[type] for type in det_types]

            for i in range(len(det_boxes)):
                o = metrics_pb2.Object()

                # The following 3 fields are used to uniquely identify a frame a prediction
                # is predicted at. Make sure you set them to values exactly the same as what
                # we provided in the raw data. Otherwise your prediction is considered as a
                # false negative.
                o.context_name = meta['seq_name']

                # The frame timestamp for the prediction. See Frame::timestamp_micros in
                # dataset.proto.
                o.frame_timestamp_micros = meta['stamp']

                # This is only needed for 2D detection or tracking tasks.
                # Set it to the camera name the prediction is for.
                #  o.camera_name = dataset_pb2.CameraName.FRONT

                # Populating box and score.
                box = label_pb2.Label.Box()
                box.center_x = det_boxes[i][0]
                box.center_y = det_boxes[i][1]
                box.center_z = det_boxes[i][2]
                box.length = det_boxes[i][3]*2
                box.width = det_boxes[i][4]*2
                box.height = det_boxes[i][5]*2
                box.heading = math.atan2(det_boxes[i][7], det_boxes[i][6])
                o.object.box.CopyFrom(box)

                # This must be within [0.0, 1.0]. It is better to filter those boxes with
                # small scores to speed up metrics computation.
                o.score = det_scores[i]

                # For tracking, this must be set and it must be unique for each tracked
                # sequence.
                #  o.object.id = 'unique object tracking ID'

                # Use correct type.
                o.object.type = det_types[i]

                # This is used for gt
                o.object.num_lidar_points_in_box = det_num_points[i]

                objects.objects.append(o)
        return objects

    def _normlize_box(self, raw_box):
        box = np.empty(8, dtype=np.float32)
        box[:3] = raw_box[:3]
        box[3:6] = raw_box[3:6] / 2
        box[6] = math.cos(raw_box[6])
        box[7] = math.sin(raw_box[6])
        return box


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


@FI.register
class WaymoObjectNSweepLoader(object):
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

        prefix, seq_id, frame_id, object_id = sweeps['prefix'], sweeps[
            'seq_id'], sweeps['frame_id'], sweeps['object_id']

        pcd_list = []
        tf_map_vehicle0 = None
        for sweep_id in range(self.nsweep):
            sweep_frame_id = max(0, frame_id-sweep_id)
            sweep_data_path = os.path.join(
                prefix, seq_id, f'{sweep_frame_id}-{object_id}.pkl')

            if not os.path.exists(f'{sweep_data_path}.gz'):
                assert(sweep_id > 0)
                continue

            pcd, tf_map_vehicle = io.load(sweep_data_path, compress=True)

            if sweep_id == 0:
                tf_map_vehicle0 = tf_map_vehicle

            if sweep_id > 0:
                tf_cur_past = rigid.between(tf_map_vehicle0, tf_map_vehicle)
                pcd = rigid.transform(tf_cur_past, pcd)
            pcd_list.append(pcd)

        points = np.concatenate(pcd_list, axis=0)

        return Pointcloud(points=points[:, :self.load_dim])
