import numpy as np

from mdet.core.geometry3d import remove_points_in_boxes, points_in_boxes


from mdet.data.transforms.transform_utils import (
    noise_per_box,
    transform_boxes,
    transform_points,
)
from mdet.utils.factory import FI

import time


@FI.register
class PcdRangeFilter(object):
    r'''
    Filter out the out of range object
    '''

    def __init__(self, box_range):
        super().__init__()
        self.box_range = np.array(box_range, dtype=np.float32)

    def __call__(self, sample, info):
        x_min = self.box_range[0]
        y_min = self.box_range[1]
        x_max = self.box_range[3]
        y_max = self.box_range[4]

        boxes = sample['anno'].boxes

        valid_indices = []
        for i, box in enumerate(boxes):
            if not(box[0] > x_min and box[0] < x_max and box[1] > y_min and box[1] < y_max):
                continue
            else:
                valid_indices.append(i)

        # update sample
        sample['anno'] = sample['anno'][valid_indices]


@FI.register
class PcdIntensityNormlizer(object):
    r'''
    Normalize the pcd intensity field to a valid range
    '''

    def __init__(self):
        super().__init__()

    def __call__(self, sample, info):
        points = sample['data']['pcd'].points
        points[:, 3] = np.tanh(points[:, 3])


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

        indice_list = points_in_boxes(points, boxes)  # [num_boxes, num_points]

        # transform points
        transform_points(points, boxes[:, :3],
                         indice_list, loc_noises, rot_noises)

        # transform box
        transform_boxes(boxes, loc_noises, rot_noises)
