from mdet.utils.factory import FI
import numpy as np
from mdet.utils.gaussian import gaussian_radius, draw_gaussian


@FI.register
class CenterAssigner(object):
    r'''
    Assign heatmap, offset, height, size, cos_theta, sin_theta, positive_indices
    offset is the box center to voxel center,
    size is in log format
    '''

    def __init__(self, point_range, grid_size, grid_reso, min_gaussian_radius=1, min_gaussian_overlap=0.5):
        # float [x_min, y_min, z_min, x_max, y_max, z_max]
        self.point_range = np.array(point_range, dtype=np.float32)
        # int [reso_x, reso_y, reso_z]
        self.grid_reso = np.array(grid_reso, dtype=np.int32)
        # float [size_x, size_y, size_z]
        self.grid_size = np.array(grid_size, dtype=np.float32)
        self.min_gaussian_radius = min_gaussian_radius
        self.min_gaussian_overlap = min_gaussian_overlap

    def __call__(self, sample, info):
        boxes = sample['gt']['boxes']
        categories = sample['gt']['categories']
        category_num = len(sample['category_id_to_name'])

        # offset
        cords_x = np.round(
            (boxes[:, 0] - self.point_range[0]) / self.grid_size[0])
        cords_y = np.round(
            (boxes[:, 1] - self.point_range[1]) / self.grid_size[1])
        cords = np.stack((cords_x, cords_y), axis=-1).astype(np.int32)

        grid_offset = self.point_range[:2] + self.grid_size[:2]/2
        center = cords * self.grid_size[:2] + grid_offset
        offset = boxes[:, :2] - center

        # positive_indices
        positive_indices = np.stack(
            (cords_y, cords_x), axis=-1).astype(np.int32)

        # height
        height = boxes[:, [2]]

        # size, in log format
        size = np.log(boxes[:, 3:6])

        # heading, in complex number format
        heading = np.stack((np.cos(boxes[:, 6]), np.sin(boxes[:, 6])), axis=-1)

        # heatmap
        heatmap = np.zeros(
            (category_num, self.grid_reso[1], self.grid_reso[0]), dtype=np.float32)
        for i in range(len(boxes)):
            box = boxes[i]
            category = categories[i]
            center = cords[i]

            l, w = box[3] / self.grid_size[0], box[4] / self.grid_size[1]
            radius = gaussian_radius((l, w), self.min_gaussian_overlap)
            radius = max(self.min_gaussian_radius, int(radius))
            draw_gaussian(heatmap[category], center, radius)

        sample['gt'].update(dict(offset=offset,
                                 height=height,
                                 size=size,
                                 heading=heading,
                                 heatmap=heatmap,
                                 positive_indices=positive_indices,
                                 ))
