import numpy as np


class Annotation3d(object):
    def __init__(self, boxes, types, scores=None, num_points=None):
        super().__init__()

        r'''
        shape: N x 8, float32
        encode: [center_x, center_y, center_z, extend_x, extend_y, extend_z, cos_alpha, sin_alpha]
        center: box center in global frame
        extend: half length and width
        rot: the box x axis relative to global x axis in counter-clock wise direction
        '''
        self.boxes = boxes

        r'''
        shape: N, int32
        type id in [0, type_num)
        '''
        self.types = types if types is not None else np.full(
            boxes.shape[0], -1, dtype=np.int32)

        r'''
        optional
        shape: N, float32
        indicate box confidence
        '''
        self.scores = scores if scores is not None else np.full(
            boxes.shape[0], 0, dtype=boxes.dtype)

        r'''
        optional
        shape: N, int32
        indicate how many points in each box
        '''
        self.num_points = num_points if num_points is not None else np.full(
            boxes.shape[0], 0, dtype=np.int32)

    def __getitem__(self, idx):
        return Annotation3d(
            boxes=self.boxes[idx],
            types=self.types[idx],
            scores=None if self.scores is None else self.scores[idx],
            num_points=None if self.num_points is None else self.num_points[idx],
        )

    def __repr__(self):
        s = 'Annotation3d{\n'
        s += f'boxes: {self.boxes},\ntypes:\n{self.types}'
        if self.scores is not None:
            s += f',\nscores:\n{self.scores}'
        if self.num_points is not None:
            s += f',\nnum_points:\n{self.num_points}'
        s += '\n}'
        return s

    def __add__(self, other):
        box_list = [self.boxes, other.boxes]
        type_list = [self.types, other.types]
        score_list = [self.scores, other.scores]
        num_points_list = [self.num_points, other.num_points]

        return Annotation3d(
            np.concatenate(box_list, axis=0),
            np.concatenate(type_list, axis=0),
            np.concatenate(score_list, axis=0),
            np.concatenate(num_points_list, axis=0),
        )

    def __iadd__(self, other):
        return self.__add__(other)
