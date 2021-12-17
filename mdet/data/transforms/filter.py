from mdet.utils.factory import FI
import numpy as np


@FI.register
class RangeFilter(object):
    r'''
    Filter out the out of range object label
    '''

    def __init__(self, point_range):
        self.point_range = np.array(point_range, dtype=np.float32)

    def __call__(self, sample, info):
        x_min = self.point_range[0]
        y_min = self.point_range[1]
        x_max = self.point_range[3]
        y_max = self.point_range[4]

        boxes = sample['gt']['boxes']

        valid_indices = []
        for i, box in enumerate(boxes):
            if not(box[0] > x_min and box[0] < x_max and box[1] > y_min and box[1] < y_max):
                continue
            else:
                valid_indices.append(i)

        sample['gt']['boxes'] = sample['gt']['boxes'][valid_indices]
        sample['gt']['categories'] = sample['gt']['categories'][valid_indices]
