import torch
import numpy as np


class Sample(dict):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def to(self, device, **kwargs):
        r'''
        Put all torch.tensor and np.ndarray elements onto device,
        if element is np.ndarray, first cast it into torch.tensor
        '''
        def wrap(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device, **kwargs)
            elif torch.is_tensor(x):
                return x.to(device, **kwargs)
            else:
                return x

        return self.__apply_all(wrap)

    def __apply_all(self, func):
        r"""
        apply func to all the tensor element
        """
        for k, v in self.items():
            self[k] = self.__apply_one(v, func)
        return self

    def __apply_one(self, item, func):
        r"""
        apply func to tensor or np.ndarray, if item is list, tuple or dict, apply it recursively
        """
        if isinstance(item, (tuple, list)):
            return [self.__apply_one(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply_one(v, func) for k, v in item.items()}
        else:
            return func(item)
