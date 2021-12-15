

from mdet.utils.factory import FI
from mdet.data.datasets.base_dataset import BaseDataset


@FI.register
class WaymoDataset(BaseDataset):
    def __init__(self, info_path, transforms=[], filter=None):
        super().__init__(info_path, transforms, filter)

    def format(self, result, output_path):
        raise NotImplementedError

    def evaluate(self, predict_path, gt_path):
        raise NotImplementedError

    def generate_gt(self, gt_path):
        raise NotImplementedError
