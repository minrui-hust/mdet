from mai.data.datasets import BaseDataset
import numpy as np
from mai.utils import FI


@FI.register
class MDet3dDataset(BaseDataset):
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

    def __init__(self, info_path, filters=None, transforms=None, codec=None):
        super().__init__(info_path, filters, transforms, codec)

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

    @classmethod
    def plot(cls, sample, show_data=True, show_anno=True, show_pred=True):
        r'''
        plot standard 3d detection sample using open3d
        '''
        from mdet.utils.viz import Visualizer

        vis = Visualizer()
        vis.add_points(sample['data']['pcd'].points)

        if show_anno and 'anno' in sample:
            box_color = np.array(
                [cls.TypePalette[type % cls.TypePalette.shape[0]] for type in sample['anno'].types])
            vis.add_box(sample['anno'].boxes,
                        box_color=box_color, box_label=None, prefix='anno')

        if show_pred and 'pred' in sample:
            mask = sample['pred'].scores > 0.3
            box_color = np.array([cls.TypePalette[(
                type+1) % cls.TypePalette.shape[0]] for type in sample['pred'].types])
            vis.add_box(sample['pred'].boxes[mask],
                        box_color=box_color[mask], box_label=None, prefix='pred')

        vis.show()
