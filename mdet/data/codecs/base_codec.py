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
        self.mode = 'train'
        self.encode_cfg = encode_cfg
        self.decode_cfg = decode_cfg
        self.loss_cfg = loss_cfg

    def set_train(self):
        self.set_mode('train')

    def set_eval(self):
        self.set_mode('eval')

    def set_infer(self):
        self.set_mode('infer')

    def set_mode(self, mode):
        assert(mode in ['train', 'eval', 'infer'])
        self.mode = mode

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

    def decode(self, output, batch=None):
        r'''
        output --> pred
        '''
        if self.mode == 'train':
            return self.decode_train(output, batch)
        elif self.mode == 'eval':
            return self.decode_eval(output, batch)
        elif self.mode == 'infer':
            return self.decode_infer(output, batch)
        else:
            raise NotImplementedError

    def decode_train(self, output, batch):
        raise NotImplementedError

    def decode_eval(self, output, batch):
        raise NotImplementedError

    def decode_infer(self, output, batch):
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
