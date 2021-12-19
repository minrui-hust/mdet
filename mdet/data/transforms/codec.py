import torch
from mdet.utils.factory import FI
import numpy as np
from mdet.utils.gaussian import gaussian_radius, draw_gaussian


class Codec(object):
    r'''
    Codec do two things:
        1. encode standard sample format into task specific format
        2. decode task specific format into standard fromat
    '''

    def __init__(self):
        super().__init__()

    def __call__(self, sample, info):
        self.encode(sample, info)

    def encode(self, sample, info):
        raise NotImplementedError

    def decode(self, **kwargs):
        raise NotImplementedError


@FI.register
class CenterPointCodec(Codec):
    r'''
    Assign heatmap, offset, height, size, cos_theta, sin_theta, positive_indices
    offset is the box center to voxel center,
    size is in log format
    '''

    def __init__(self, point_range, grid_size, grid_reso, min_gaussian_radius=1, min_gaussian_overlap=0.5):
        super().__init__()

        # float [x_min, y_min, z_min, x_max, y_max, z_max]
        self.point_range = np.array(point_range, dtype=np.float32)
        # int [reso_x, reso_y, reso_z]
        self.grid_reso = np.array(grid_reso, dtype=np.int32)
        # float [size_x, size_y, size_z]
        self.grid_size = np.array(grid_size, dtype=np.float32)
        self.min_gaussian_radius = min_gaussian_radius
        self.min_gaussian_overlap = min_gaussian_overlap

    def encode(self, sample, info):
        boxes = sample['anno'].boxes
        types = sample['anno'].types
        type_num = len(sample['type_name'])

        # offset
        cords_x = np.floor(
            (boxes[:, 0] - self.point_range[0]) / self.grid_size[0])
        cords_y = np.floor(
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
            (type_num, self.grid_reso[1], self.grid_reso[0]), dtype=np.float32)
        for i in range(len(boxes)):
            box = boxes[i]
            type = types[i]
            center = cords[i]

            # skip object out of bound
            if not(center[0] >= 0 and center[0] < self.grid_reso[0] and
                    center[1] >= 0 and center[1] < self.grid_reso[1]):
                continue

            l, w = box[3] / self.grid_size[0], box[4] / self.grid_size[1]
            radius = gaussian_radius((l, w), self.min_gaussian_overlap)
            radius = max(self.min_gaussian_radius, int(radius))
            draw_gaussian(heatmap[type], center, radius)

        sample['pcd'] = sample['pcd'].points

        sample['anno'] = dict(offset=offset,
                              height=height,
                              size=size,
                              heading=heading,
                              positive_indices=positive_indices,
                              heatmap=heatmap,
                              types=types,
                              )

    def decode(self, boxes, cords):
        r'''
        decode the encoded boxes to standard format
        input:
            boxes: in shape B x N x 8, encoded as
            [offset_x, offset_y, height, size_x, size_y, size_z, cos_theta, sin_theta] 
            cords: in shape B x N x 2 or B x N, the box coordinates in output feature map, encoded as
            [cord_x, cord_y]
        output:
            shape B x N x 7, in standard format, which is
            [center_x, center_y, center_z, size_x, size_y, size_z, rot]
        '''

        grid_offset = self.point_range[:2] + self.grid_size[:2]/2
        grid_offset = torch.from_numpy(grid_offset).to(boxes.device)
        grid_size = torch.from_numpy(self.grid_size[:2]).to(boxes.device)

        # if is B x N, convert to B x N x 2
        if cords.dim() == 2:
            cords_y = cords//grid_size[0]
            cords_x = cords % grid_size[0]
            cords = torch.stack([cords_x, cords_y], dim=-1)

        grid_center = cords * grid_size + grid_offset

        decoded_box_xy = boxes[..., :2] + grid_center

        decoded_box_z = boxes[..., [2]]

        decoded_box_size = torch.exp(boxes[..., 3:6])

        decoded_box_rot = torch.atan2(boxes[..., [7]], boxes[..., [6]])

        decoded_box = torch.cat([
            decoded_box_xy,
            decoded_box_z,
            decoded_box_size,
            decoded_box_rot
        ], dim=-1)

        return decoded_box
