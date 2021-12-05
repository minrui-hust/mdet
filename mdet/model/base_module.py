import torch.nn as nn


class BaseModule(nn.Module):
    r'''
    Extention of torch.nn.Module, support different mode('train', 'val', 'test', 'infer')
    '''

    def __init__(self):
        super().__init__()
        self.mode = 'train'

    def set_train(self):
        self.set_mode('train')

    def set_eval(self):
        self.set_mode('val')

    def set_test(self):
        self.set_mode('test')

    def set_infer(self):
        self.set_mode('infer')

    def set_mode(self, mode):
        if mode in ['train']:
            self.train()
        if mode in ['val', 'test', 'infer']:
            self.eval()

        for m in self.children():
            if isinstance(m, BaseModule):
                m.set_mode(mode)

    def forward(self, *args, **kwargs):
        if self.mode == 'train':
            return self.forward_train(*args, **kwargs)
        elif self.mode == 'val':
            return self.forward_val(*args, **kwargs)
        elif self.mode == 'test':
            return self.forward_test(*args, **kwargs)
        elif self.mode == 'infer':
            return self.forward_export(*args, **kwargs)
        else:
            raise NotImplementedError

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError

    def forward_val(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_test(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)

    def forward_export(self, *args, **kwargs):
        return self.forward_train(*args, **kwargs)
