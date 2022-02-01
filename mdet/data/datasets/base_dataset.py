import numpy as np
from torch.utils.data import Dataset as TorchDataset

from mdet.data.sample import Sample
from mdet.utils.factory import FI
import mdet.utils.io as io


class MDetDataset(TorchDataset):
    r'''
    Base class of all dataset in MDet
    '''

    def __init__(self, info_path, transforms=[], codec=None, filter=None):
        super().__init__()

        self.info_path = info_path

        self.transforms = [FI.create(cfg) for cfg in transforms]

        self.codec = FI.create(codec)

        self.filter = FI.create(filter)

        # load sample info
        sample_infos = io.load(self.info_path, format='pkl')
        self.sample_infos = [info for i, info in enumerate(
            sample_infos) if (self.filter is None or self.filter(info, i))]

    def __len__(self):
        return len(self.sample_infos)

    def __getitem__(self, idx):
        info = self.sample_infos[idx]
        if isinstance(info, list):
            sample_list = []
            for i in info:
                sample = Sample()
                self.process(sample, i)
            return sample_list
        else:
            sample = Sample()
            self.process(sample, info)
            return sample

    def process(self, sample, info):
        self.load(sample, info)
        self.transform(sample, info)
        self.encode(sample, info)

    def load(self, sample, info):
        r'''
        Need override by derived class
        '''
        raise NotImplementedError

    def transform(self, sample, info):
        if self.transforms:
            for t in self.transforms:
                t(sample, info)

    def encode(self, sample, info):
        r'''
        encode standard sample format to task specified
        '''
        if self.codec:
            self.codec.encode(sample, info)

    def plot(self, sample):
        raise NotImplementedError

    def format(self, result, pred_path=None, gt_path=None):
        r'''
        Format results into dataset specific format for evaluation and submission
        '''
        raise NotImplementedError

    def evaluate(self, predict_path, gt_path=None):
        r'''
        Evaluate  predictions
        '''
        raise NotImplementedError


class MDet3dDataset(MDetDataset):
    r'''
    Base class of all 3d detection dataset
    Sample format:{
                   'data':{
                       'pcd': Pointcloud
                   }
                   'anno': Annotation3d, optional, filled by annotation
                   'pred': Annotation3d, optional, filled by prediction
                   'meta':{
                       'sample_name': str
                       'type_name': list of str
                   }
                  }
    '''
    TypePalette = np.array([[0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.9]])

    def __init__(self, info_path, transforms=None, codec=None, filter=None):
        super().__init__(info_path, transforms, codec, filter)

    def load(self, sample, info):
        self.load_meta(sample, info)
        self.load_data(sample, info)
        self.load_anno(sample, info)

    def load_meta(self, sample, info):
        r'''
        load meta data into sample
        '''
        raise NotImplementedError

    def load_data(self, sample, info):
        r'''
        load data into sample, store as standard format
        '''
        raise NotImplementedError

    def load_anno(self, sample, info):
        r'''
        load dataset specific annotation , return standard 3d annotation
        '''
        raise NotImplementedError

    def plot(self, sample, show_data=True, show_anno=True, show_pred=True):
        r'''
        plot standard 3d detection sample using open3d
        '''
        from mdet.utils.viz import Visualizer

        vis = Visualizer()
        vis.add_points(sample['data']['pcd'].points)

        type2label = sample['meta']['type2label']

        if show_anno and 'anno' in sample:
            box_color = [self.TypePalette[type2label[type] %
                                          self.TypePalette.shape[0]] for type in sample['anno'].types]
            box_label = [str(type_id) for type_id in sample['anno'].types]
            vis.add_box(sample['anno'].boxes,
                        box_color=box_color, box_label=None, prefix='anno')

        if show_pred and 'pred' in sample:
            mask = sample['pred'].scores > 0.3
            box_color = [self.TypePalette[(type2label[type]+1) %
                                          self.TypePalette.shape[0]] for type in sample['pred'].types]
            box_label = [str(type_id) for type_id in sample['pred'].types]
            vis.add_box(sample['pred'].boxes[mask],
                        box_color=box_color[mask], box_label=None, prefix='pred')

        vis.show()


class MDet2dDataset(MDetDataset):
    r'''
    Base class of 2d detection dataset
    '''

    def __init__(self, info_path, transforms=None, codec=None, filter=None):
        super().__init__(info_path, transforms, codec, filter)

    def load(self, sample, info):
        pass

    def load_img(self, img_path):
        raise NotImplementedError

    def load_anno(self, anno_path):
        raise NotImplementedError
