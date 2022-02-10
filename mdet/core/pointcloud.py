import numpy as np
from mdet.core.box_np_ops import points_in_boxes


class Pointcloud(object):
    def __init__(self, points):
        super().__init__()

        r'''
        shape: N x F, float32
        encode: [x, y, z, other feature]
        '''
        self.points = points

    def __add__(self, other):
        point_list = [self.points, other.points]
        return Pointcloud(np.concatenate(point_list, axis=0))

    def __iadd__(self, other):
        return self.__add__(other)
