import torch.utils.data as pdata
from mdet.utils.factory import FI
import mdet.utils.io as io


@FI.register
class BaseDataset(pdata.Dataset):
    def __init__(self, info_path, transforms_cfg, filter_cfg):
        super().__init__()

        self.info_path = info_path
        if transforms_cfg:
            self.transforms = [FI.create(cfg) for cfg in transforms_cfg]
        else:
            self.transforms = None

        if filter_cfg:
            self.filter = FI.create(filter_cfg)
        else:
            self.filter = lambda x: True

        sample_infos = io.load(self.info_path, format='pkl')
        self.sample_infos = [
            info for info in sample_infos if self.filter(info)]

    def __len__(self):
        return len(self.sample_infos)

    def __getitem__(self, idx):
        info = self.sample_infos[idx]
        sample = {}
        self.__pre_transform(sample, info)
        self.__do_transform(sample, info)
        self.__post_transform(sample, info)
        return sample

    def format(self, result, output_path):
        r'''
        Format results into specific format for evaluation.
        '''
        pass

    def evaluate(self, predict_path, gt_path):
        r'''
        Evaluate  predictions
        '''
        pass

    def generate_gt(self, gt_path):
        r'''
        generate gt file for evaluation
        '''
        pass

    def __pre_transform(self, sample, info):
        r'''
        Initialization before transforms, overload by inherits
        '''
        pass

    def __do_transform(self, sample, info):
        if self.transforms:
            for t in self.transforms:
                t(sample, info)

    def __post_transform(self, sample, info):
        r'''
        Cleaning after transforms, overload by inherits
        '''
        pass
