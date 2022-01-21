import os
import numpy as np
from mdet.utils.factory import FI
import mdet.utils.io as io
from mdet.core.annotation import Annotation3d
from mdet.core.pointcloud import Pointcloud
from mdet.core.box_np_ops import box_bev_corner, box_collision_test


class BatchSampler:
    """sampling a batch of ground truths boxes of specific label.

    Args:
        label (int): label id
        sample_list (list[dict]): List of all samples.
        shuffle (bool): Whether to shuffle indices. Default: False.
    """

    def __init__(self, label, sampled_list, shuffle=True):
        self.label = label
        self.sampled_list = sampled_list
        self.shuffle = shuffle

        self.total_num = len(sampled_list)
        self.indices = np.arange(self.total_num)
        self.cur_idx = 0

        self.reset()

    def sample(self, num):
        if self.cur_idx + num >= self.total_num:
            self.reset()

        indices = self.indices[self.cur_idx:self.cur_idx + num]
        self.cur_idx += num

        return [self.sampled_list[i] for i in indices]

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.cur_idx = 0


@FI.register
class GroundTruthSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        sample_groups (dict): Sampled classes and numbers.
        classes (list): specific how type mapping to labels.
        filter: filter config to downsample database
    """

    def __init__(self,
                 info_path,
                 sample_groups,
                 labels,
                 pcd_loader,
                 filter=None):
        super().__init__()
        self.info_path = info_path
        self.sample_groups = sample_groups
        self.labels = labels
        self.pcd_loader = FI.create(pcd_loader)
        self.filter = FI.create(filter)

        self.type2label = {}  # map type id to lable name
        self.labelname2id = {}
        for label_id, (label_name, type_list) in enumerate(self.labels):
            for type in type_list:
                self.type2label[type] = label_name
            self.labelname2id[label_name] = label_id

        # load infos
        db_infos = io.load(info_path)

        # filter info and orgnize by label but raw type
        self.label_info_list = {label: [] for label, _ in self.labels}
        for cat, info_list in db_infos.items():
            if cat not in self.type2label:  # this cat is not interested
                continue

            cat_info = [
                info for info in info_list
                if not self.filter or self.filter(info)
            ]
            self.label_info_list[self.type2label[cat]].extend(cat_info)

        self.sampler_dict = {}
        for label, infos in self.label_info_list.items():
            self.sampler_dict[label] = BatchSampler(label, infos, shuffle=True)

    def sample_all(self, gt_anno):
        """Sampling all categories of bboxes.

        Args:
            gt_anno(Annotation3d): original gt annotations

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.
                - sample_anno(Annotation3d): sampled gt annotations
                - sample_pcd(PointCloud): sampled gt pointcloud
        """
        sampled_num_dict = {}
        for label_name, max_sample_num in self.sample_groups.items():
            label_id = self.labelname2id[label_name]
            sampled_num = int(max_sample_num -
                              np.sum([n == label_id
                                      for n in gt_anno.types]))
            sampled_num_dict[label_name] = sampled_num

        samples = []
        sampled_box_list = []
        avoid_coll_boxes = gt_anno.boxes

        for class_name, sampled_num in sampled_num_dict.items():
            if sampled_num > 0:
                class_samples = self.sample_class(class_name, sampled_num,
                                                  avoid_coll_boxes)
                if len(class_samples) > 0:
                    samples += class_samples
                    sampled_class_boxes = np.stack(
                        [info['box'] for info in class_samples], axis=0)

                    sampled_box_list.append(sampled_class_boxes)
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_class_boxes], axis=0)

        sampled_points_list = [self.pcd_loader(
            info['sweeps']).points for info in samples]
        sampled_type_list = [
            self.labelname2id[self.type2label[info['type']]] for info in samples]
        sampled_num_points_list = [info['num_points'] for info in samples]

        if len(sampled_box_list) > 0:
            sampled_boxes = np.concatenate(sampled_box_list, axis=0)
            sampled_types = np.stack(sampled_type_list, axis=0)
            sampled_num_points = np.stack(sampled_num_points_list, axis=0)
            sampled_points = np.concatenate(sampled_points_list, axis=0)
        else:
            sampled_boxes = np.empty((0, 8), dtype=np.float32)
            sampled_types = np.empty((0, ), dtype=np.int32)
            sampled_num_points = np.empty((0, ), dtype=np.int32)
            sampled_points = np.empty(
                (0, self.pcd_loader.load_dim), dtype=np.float32)

        return dict(
            sample_anno=Annotation3d(boxes=sampled_boxes,
                                     types=sampled_types,
                                     num_points=sampled_num_points),
            sample_pcd=Pointcloud(points=sampled_points),
        )

    def sample_class(self, label, num, gt_boxes):
        """Sampling specific categories of bounding boxes.

        Args:
            label (int): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        samples = self.sampler_dict[label].sample(num)
        sampled_boxes = np.stack([info['box'] for info in samples], axis=0)

        coll_mat = box_collision_test(sampled_boxes, gt_boxes)

        valid_indices = []
        for i in range(len(samples)):
            if not coll_mat[i].any():
                valid_indices.append(i)

        remain_sampled_boxes = sampled_boxes[valid_indices]

        coll_mat = box_collision_test(remain_sampled_boxes,
                                      remain_sampled_boxes)

        remained_num = remain_sampled_boxes.shape[0]

        diag = np.arange(remained_num)
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(remained_num):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(samples[valid_indices[i]])
        return valid_samples


@FI.register
class FilterByDifficulty(object):
    r'''Filter ground truths by difficulties.
    '''

    def __init__(self, difficulties_to_remove=[]):
        super().__init__()
        self.difficulties_to_remove = difficulties_to_remove

    def __call__(self, info):
        return info['difficulty'] not in self.difficulties_to_remove


@FI.register
class FilterByNumpoints(object):
    r'''
    filter ground truths by num points
    '''

    def __init__(self, min_num_points=1):
        super().__init__()
        self.min_num_points = min_num_points

    def __call__(self, info):
        return info['num_points'] >= self.min_num_points