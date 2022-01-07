import torch
import torch.nn as nn


class BaseCodec(object):
    r'''
    What codec does:
        1. encode standard sample into task specific format,
        2. decode task specific output into standard fromat(that is same as annotation)
        3. calc loss given output and gt
        5. define the collator to collate the encoded sample
        4. and also a plot method to viz encoded sample
    '''

    def __init__(self,
                 encode_cfg={'encode_data': True, 'encode_anno': True},
                 decode_cfg={},
                 loss_cfg={},
                 ):
        super().__init__()
        self.encode_cfg = encode_cfg
        self.decode_cfg = decode_cfg
        self.loss_cfg = loss_cfg

    def encode(self, sample, info):
        r'''
        data --> input
        anno --> gt
        after encode, sample would be like this:
        {'input', 'gt', 'data', 'anno', 'meta'}
        '''
        if self.encode_cfg['encode_data']:
            self.encode_data(sample, info)
        if self.encode_cfg['encode_anno']:
            self.encode_anno(sample, info)

    def decode(self, output, batch):
        r'''
        output --> pred
        '''
        raise NotImplementedError

    def loss(self, output, batch):
        r'''
        codec encodes the gt, so it knows how to calc loss
        '''
        raise NotImplementedError

    def get_collater(self):
        r'''
        codec encodes the sample, so it knows how to collate the sample
        '''
        raise NotImplementedError

    def plot(self, sample, show_input=True, show_gt=True, show_output=True):
        r'''
        codec know how to visualize sample it encodes
        '''
        raise NotImplementedError

    def encode_data(self, sample, info):
        r'''
        encode data to model input
        '''
        raise NotImplementedError

    def encode_anno(self, sample, info):
        r'''
        encode anno to gt for training and evaluation
        '''
        raise NotImplementedError

    def get_export_info(self, batch):
        raise NotImplementedError
