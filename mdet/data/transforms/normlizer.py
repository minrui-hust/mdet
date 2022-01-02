from mdet.utils.factory import FI
import numpy as np


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
