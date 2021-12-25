import torch
import numpy as np


class Sample(dict):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def to(self, device, **kwargs):
        r'''
        Put all torch.tensor elements onto device,
        '''
        def wrap(x):
            if torch.is_tensor(x):
                return x.to(device, **kwargs)
            else:
                return x

        return self.__apply_all(wrap)

    def select(self, elements):
        return Sample(**{key: self[key] for key in elements if key in self})

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
