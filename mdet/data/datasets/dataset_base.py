import torch.utils.data as pdata
from components.component_factory import ComponentFactory
from data.sample import Sample
import pickle

cf = ComponentFactory()


@cf.registerComponent()
class Dataset(pdata.Dataset):
    def __init__(self, info_path, transforms, filter=lambda x: True):
        super().__init__()

        self.info_path = info_path
        self.transforms = transforms
        self.filter = filter

        sample_infos = self._load_info(self.info_path)
        self.sample_infos = [
            info for info in sample_infos if self.filter(info)]

    def __len__(self):
        return len(self.sample_infos)

    def __getitem__(self, idx):
        sample = Sample(info=self.sample_infos[idx])
        self._pre_transform(sample)
        self._do_transform(sample)
        self._post_transform(sample)
        return sample

    def format(self, results, pickle_path=None, submission_path=None):
        r'''
        Format results into specific format for evaluation
        '''
        pass

    def evaluate(self):
        r'''
        Evaluate  results formated by format
        '''
        pass

    def _load_info(self, info_path):
        with open(info_path, 'rb') as fin:
            return pickle.load(fin)

    def _pre_transform(self, sample):
        r'''
        Initialization before transforms, overload by inherits
        '''
        pass

    def _do_transform(self, sample):
        for t in self.transforms:
            t(sample)

    def _post_transform(self, sample):
        r'''
        Cleaning after transforms, overload by inherits
        '''
        pass
