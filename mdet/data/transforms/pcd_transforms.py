import time

import numpy as np

from mdet.core.geometry2d import box_collision_test
from mdet.core.geometry3d import (
    points_in_boxes,
    remove_points_in_boxes,
    remove_points_in_boxes,
)
from mdet.core.pointcloud import Pointcloud
from mdet.data.transforms.transform_utils import (
    noise_per_box,
    transform_boxes,
    transform_points,
)
from mai.utils import FI


@FI.register
class PcdRangeFilter(object):
    r'''
    Filter out the out of range object
    '''

    def __init__(self, point_range, margin):
        super().__init__()
        self.point_range = point_range
        self.margin = margin

    def __call__(self, sample, info):
        x_min = self.point_range[0]
        y_min = self.point_range[1]
        x_max = self.point_range[3]
        y_max = self.point_range[4]

        points = sample['data']['pcd'].points

        points_mask = (points[:, 0] > x_min) & (points[:, 0] < x_max) & (
            points[:, 1] > y_min) & (points[:, 1] < y_max)

        # update points
        sample['data']['pcd'].points = points[points_mask]

        boxes = sample['anno'].boxes

        boxes_mask = (boxes[:, 0] > x_min+self.margin) & (boxes[:, 0] < x_max - self.margin) & (
            boxes[:, 1] > y_min+self.margin) & (boxes[:, 1] < y_max-self.margin)

        # update anno
        sample['anno'] = sample['anno'][boxes_mask]


@FI.register
class PointNumFilter(object):
    r'''
    Filter out the box with too less points
    '''

    def __init__(self, groups={}):
        super().__init__()
        self.groups = groups

    def __call__(self, sample, info):
        labels_name = sample['meta']['labels_name']
        type2label = sample['meta']['type2label']

        label_min_points = {}
        type_min_points = {}

        for label_name, min_points in self.groups.items():
            label = labels_name.index(label_name)
            label_min_points[label] = min_points

        for type, type_label in type2label.items():
            type_min_points[type] = label_min_points[type_label]

        num_points = sample['anno'].num_points
        types = sample['anno'].types

        boxes_mask = []
        for i in range(len(types)):
            point_num = num_points[i]
            type = types[i]
            if point_num >= type_min_points[type]:
                boxes_mask.append(i)

        sample['anno'] = sample['anno'][boxes_mask]


@FI.register
class PointNumFilterV2(object):
    r'''
    Filter out the box with too less points
    '''

    def __init__(self, groups={}):
        super().__init__()
        self.groups = groups

    def __call__(self, sample, info):
        num_points = sample['anno'].num_points
        types = sample['anno'].types

        boxes_mask = []
        for i in range(len(types)):
            point_num = num_points[i]
            type = types[i]
            if isinstance(self.groups, dict):
                min_num = self.groups[type]
            else:
                min_num = self.groups
            if point_num >= min_num:
                boxes_mask.append(i)

        sample['anno'] = sample['anno'][boxes_mask]


@FI.register
class AnnoRetyper(object):
    r'''
    assign new type id
    '''

    def __init__(self, retype=None):
        super().__init__()
        self.retype = FI.create(retype)

    def __call__(self, sample, info):
        if self.retype:
            sample['anno'] = self.retype.retype_anno(sample['anno'])
            sample['meta']['type_new_to_raw'] = self.retype.get_type_map()


@FI.register
class PcdIntensityNormlizer(object):
    r'''
    Normalize the pcd intensity field to a valid range
    '''

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def __call__(self, sample, info):
        points = sample['data']['pcd'].points
        points[:, 3] = np.tanh(points[:, 3]*self.scale)


@FI.register
class PcdShuffler(object):
    r'''
    Shuffle the points in pcd
    '''

    def __init__(self):
        super().__init__()

    def __call__(self, sample, info):
        sample['data']['pcd'].points = np.random.permutation(
            sample['data']['pcd'].points)


@FI.register
class PcdGlobalTransform(object):
    r'''
    Apply global translation, rotation and scaling

    Args:
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
    '''

    def __init__(self, translation_std=[0, 0, 0], rot_range=[-0.78539816, 0.78539816], scale_range=[0.95, 1.05]):
        super().__init__()

        self.translation_std = np.array(translation_std, dtype=np.float32)
        self.rot_range = rot_range
        self.scale_range = scale_range

    def __call__(self, sample, info):
        self._scale(sample)
        self._rotate(sample)
        self._translate(sample)

    def _scale(self, sample):
        scale_factor = np.random.uniform(
            self.scale_range[0], self.scale_range[1])

        # scale points
        sample['data']['pcd'].points[:, :3] *= scale_factor

        # scale gt boxes
        sample['anno'].boxes[:, :3] *= scale_factor

    def _rotate(self, sample):
        alpha = np.random.uniform(self.rot_range[0], self.rot_range[1])
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        rotm = np.array(
            [[cos_alpha, -sin_alpha], [sin_alpha, cos_alpha]], dtype=np.float32)

        # rotate points
        sample['data']['pcd'].points[:,
                                     :2] = sample['data']['pcd'].points[:, :2] @ rotm

        #  rotate boxes center
        sample['anno'].boxes[:, :2] = sample['anno'].boxes[:, :2] @ rotm

        # rotate boxes rotation
        sample['anno'].boxes[:, 6:] = sample['anno'].boxes[:, 6:] @ rotm

    def _translate(self, sample):
        translation = np.random.normal(scale=self.translation_std, size=3)

        # translate points
        sample['data']['pcd'].points[:, :3] += translation

        # translate gt boxes
        sample['anno'].boxes[:, :3] += translation


@FI.register
class PcdMirrorFlip(object):
    r'''
    random mirror(left-right), and flip(up-down)
    '''

    def __init__(self, mirror_prob=0.5, flip_prob=0.5):
        super().__init__()

        self.mirror_prob = mirror_prob
        self.flip_prob = flip_prob

    def __call__(self, sample, info):
        self._mirror(sample)
        self._flip(sample)

    def _mirror(self, sample):
        if self.mirror_prob < np.random.rand():
            return

        # mirror points
        sample['data']['pcd'].points[:, 1] = - \
            sample['data']['pcd'].points[:, 1]

        # mirror boxes
        sample['anno'].boxes[:, 1] = -sample['anno'].boxes[:, 1]
        sample['anno'].boxes[:, 7] = -sample['anno'].boxes[:, 7]

    def _flip(self, sample):
        if self.flip_prob < np.random.rand():
            return

        # flip points
        sample['data']['pcd'].points[:, 0] = - \
            sample['data']['pcd'].points[:, 0]

        # flip boxes
        sample['anno'].boxes[:, 0] = -sample['anno'].boxes[:, 0]
        sample['anno'].boxes[:, 6] = -sample['anno'].boxes[:, 6]


@FI.register
class PcdObjectSampler(object):
    r'''
    Random sample ground truth
    '''

    def __init__(self, db_sampler):
        super().__init__()
        self.db_sampler = FI.create(db_sampler)

    def __call__(self, sample, info):
        sampled = self.db_sampler.sample_all(sample['anno'])

        # merge sampled and original
        sample['anno'] += sampled['sample_anno']

        # remove points in sampled boxes
        sample['data']['pcd'].points = remove_points_in_boxes(
            sample['data']['pcd'].points, sampled['sample_anno'].boxes)

        # add points in sampled boxes
        sample['data']['pcd'] += sampled['sample_pcd']


@FI.register
class PcdObjectSamplerV2(object):
    def __init__(self, db_sampler, sample_groups):
        self.sampler = FI.create(db_sampler)
        self.sample_groups = sample_groups

    def __call__(self, sample, info):
        anno = sample['anno']

        gt_boxes = anno.boxes
        gt_types = anno.types
        gt_points = sample['data']['pcd'].points

        typed_sample_num = {}
        for type, max_num in self.sample_groups.items():
            type_indice = (gt_types == type).nonzero()
            typed_sample_num[type] = max(0, max_num - len(type_indice))

        # do sample according to typed_sample_num
        sampled_anno, sampled_points_list = self.sampler.sample(
            typed_sample_num)

        # do nms, keep valid samples
        sampled_boxes = sampled_anno.boxes
        total_boxes = np.concatenate([gt_boxes, sampled_boxes], axis=0)
        total_boxes = total_boxes[:, [0, 1, 3, 4, 6, 7]]
        collision_matrix = box_collision_test(total_boxes, total_boxes)
        valid_sample_id_list = []
        for id in range(len(gt_boxes), len(total_boxes)):
            # this box should be supressed
            if collision_matrix[:id, id].any():
                # clear id-th row and col, indicate this boxes ommited
                collision_matrix[id, :] = 0
                collision_matrix[:, id] = 0
            else:  # this box should be selected
                valid_sample_id_list.append(id-len(gt_boxes))

        sampled_anno = sampled_anno[valid_sample_id_list]
        sampled_points = np.concatenate(
            [sampled_points_list[id] for id in valid_sample_id_list], axis=0)

        # merge
        # remove points in sampled boxes first
        sample['data']['pcd'].points = remove_points_in_boxes(
            gt_points, sampled_anno.boxes)

        sample['data']['pcd'] += Pointcloud(points=sampled_points)
        sample['anno'] += sampled_anno


@FI.register
class PcdLocalTransform(object):
    r'''
    Apply local transform to each ground truth object
    '''

    def __init__(self, translation_std=[0, 0, 0], rot_range=[-0.78539816, 0.78539816], num_try=50):
        super().__init__()

        self.translation_std = np.array(translation_std, dtype=np.float32)
        self.rot_range = rot_range
        self.num_try = num_try

    def __call__(self, sample, info):
        boxes = sample['anno'].boxes
        points = sample['data']['pcd'].points

        num_boxes = boxes.shape[0]

        loc_noises = np.random.normal(
            scale=self.translation_std, size=[num_boxes, self.num_try, 3]).astype(np.float32)
        angle_noises = np.random.uniform(
            self.rot_range[0], self.rot_range[1], size=[num_boxes, self.num_try]).astype(np.float32)
        rot_noises = np.stack(
            [np.cos(angle_noises), np.sin(angle_noises)], axis=-1)

        noise_indices = noise_per_box(boxes, loc_noises, rot_noises)[
            :, np.newaxis, np.newaxis]

        loc_noises = np.take_along_axis(loc_noises, np.broadcast_to(
            noise_indices, (num_boxes, 1, 3)), axis=1).squeeze(1)
        rot_noises = np.take_along_axis(rot_noises, np.broadcast_to(
            noise_indices, (num_boxes, 1, 2)), axis=1).squeeze(1)

        indice_list_raw, kdt = points_in_boxes(points, boxes, return_kdt=True)

        # transform boxes
        transform_boxes(boxes, loc_noises, rot_noises)
        indice_list_new = points_in_boxes(points, boxes, kdt=kdt)

        # get trasformed boxes' points
        box_points_list = [points[idx] for idx in indice_list_raw]
        transform_points(box_points_list, boxes[:, :3], loc_noises, rot_noises)
        boxes_points = np.concatenate(box_points_list, axis=0)

        # remove points in boxes(extract back ground points)
        del_indice = np.concatenate(indice_list_raw + indice_list_new)
        bg_points = np.delete(points, del_indice, axis=0)

        # concatenate back ground points and boxes' points
        sample['data']['pcd'].points = np.concatenate(
            [bg_points, boxes_points], axis=0)
