import torch
import torch.nn.functional as F
from mdet.utils.misc import is_list_of


class CollateOperator(object):
    def __init__(self, collator):
        super().__init__()
        self.collator = collator

    def __call__(self, data, name_str, batch_info, is_collate=True):
        if is_collate:
            return self.collate(data, name_str, batch_info)
        else:
            return self.decollate(data, name_str, batch_info)

    def collate(self, data_list, name_str, batch_info):
        raise NotImplementedError

    def decollate(self, data_batch, name_str, batch_info):
        raise NotImplementedError


class CollateOperatorUnique(CollateOperator):
    def __init__(self, collator):
        super().__init__(collator)

    def collate(self, data_list, name_str, batch_info):
        if not data_list:
            return None
        return data_list[0]

    def decollate(self, data_batch, name_str, batch_info):
        return [data_batch.copy() for _ in range(batch_info['size'])]


class CollateOperatorAppend(CollateOperator):
    def __init__(self, collator):
        super().__init__(collator)

    def collate(self, data_list, name_str, batch_info):
        return data_list

    def decollate(self, data_batch, name_str, batch_info):
        assert(isinstance(data_batch, list))
        return data_batch


class CollateOperatorCat(CollateOperator):
    def __init__(self, collator, dim=0, pad_cfg=None, inc_func=None):
        super().__init__(collator)
        self.dim = dim
        self.inc_func = inc_func
        self.pad_cfg = pad_cfg

    def collate(self, data_list, name_str, batch_info):
        r'''
        pad-->inc->cat
        '''
        if not data_list:
            return None

        new_data_list = []
        acc_list = [0]
        sections = []
        for data in data_list:
            inc = 0 if self.inc_func is None else self.inc_func(data)
            if self.pad_cfg is not None:
                data = F.pad(data, **self.pad_cfg)
            new_data_list.append(data + acc_list[-1])
            acc_list.append(acc_list[-1]+inc)
            sections.append(data.size(self.dim))

        # record info for decollate
        batch_info[f'{name_str}.sections'] = sections
        batch_info[f'{name_str}.accumulations'] = acc_list[:-1]

        return torch.cat(new_data_list, dim=self.dim)

    def decollate(self, data_batch, name_str, batch_info):
        sections = batch_info.get(f'{name_str}.sections', None)
        accumulations = batch_info.get(f'{name_str}.accumulations', None)

        # 1. do split, which reverse the cat operation
        assert(sections is not None)
        data_list = list(data_batch.split(sections, self.dim))

        # 2. decrease
        if accumulations is not None:
            for i, data in enumerate(data_list):
                data_list[i] = data-accumulations[i]

        # 3. remove padding
        if self.pad_cfg is not None:
            pad_size_list = self.pad_cfg['pad']
            for i, data in enumerate(data_list):
                for j, pad_size in enumerate(pad_size_list):
                    dim = -1 - int(j/2)  # reverse order
                    is_front = bool((j+1) % 2)
                    start = pad_size if is_front else 0
                    length = data.size(dim) - pad_size
                    data = data.narrow(dim, start, length)
                data_list[i] = data

        return data_list


class CollateOperatorStack(CollateOperator):
    def __init__(self, collator, dim=0, inc_func=None):
        super().__init__(collator)
        self.dim = dim
        self.inc_func = inc_func

    def collate(self, data_list, name_str, batch_info):
        if not data_list:
            return None

        new_data_list = []
        acc_list = [0]
        for data in data_list:
            inc = 0 if self.inc_func is None else self.inc_func(data)
            new_data_list.append(data + acc_list[-1])
            acc_list.append(acc_list[-1]+inc)

        # record info for decollate
        batch_info[f'{name_str}.accumulations'] = acc_list[:-1]

        return torch.stack(new_data_list, dim=self.dim)

    def decollate(self, data_batch, name_str, batch_info):
        accumulations = batch_info.get(f'{name_str}.accumulations', None)

        # 1. do chunk, which reverse the stack operation
        data_list = list(data_batch.unbind(self.dim))

        # 2. decrease
        if accumulations is not None:
            for i, data in enumerate(data_list):
                data_list[i] = data-accumulations[i]

        return data_list


class CollateOperatorRecusive(CollateOperator):
    def __init__(self, collator):
        super().__init__(collator)

    def collate(self, data_list, name_str, batch_info):
        if not data_list:
            return None
        if is_list_of(data_list, list):
            return self.__collate_list(data_list, name_str, batch_info)
        elif is_list_of(data_list, tuple):
            return self.__collate_tuple(data_list, name_str, batch_info)
        elif is_list_of(data_list, dict):
            return self.__collate_dict(data_list, name_str, batch_info)
        else:
            raise NotImplementedError

    def decollate(self, data_batch, name_str, batch_info):
        if not data_batch:
            return None
        if isinstance(data_batch, list):
            return self.__decollate_list(data_batch, name_str, batch_info)
        elif isinstance(data_batch, tuple):
            return self.__decollate_tuple(data_batch, name_str, batch_info)
        elif isinstance(data_batch, dict):
            return self.__decollate_dict(data_batch, name_str, batch_info)
        else:
            raise NotImplementedError

    def __collate_list(self, data_list, name_str, batch_info):
        # init the list
        batch = [[] for _ in range(len(data_list[0]))]

        # collect samples
        for sample in data_list:
            for idx, value in enumerate(sample):
                batch[idx].append(value)

        # do collate
        for idx, value_list in enumerate(batch):
            child_name_str = f'{name_str}.{idx}'
            collate_op = self.collator.get_collate_op(
                value_list, child_name_str)
            batch[idx] = collate_op(value_list, child_name_str, batch_info)

        return batch

    def __collate_tuple(self, data_list, name_str, batch_info):
        return tuple(self.__collate_list(data_list, name_str, batch_info))

    def __collate_dict(self, data_list, name_str, batch_info):
        # init the dict
        batch = {}
        keys = data_list[0].keys()
        for key in keys:
            batch[key] = []

        # collect samples
        for sample in data_list:
            for key, value in sample.items():
                batch[key].append(value)

        # do collate
        for key, value_list in batch.items():
            child_name_str = f'{name_str}.{key}'
            collate_op = self.collator.get_collate_op(
                value_list, child_name_str)
            batch[key] = collate_op(value_list, child_name_str, batch_info)

        return batch

    def __decollate_list(self, data_batch, name_str, batch_info):
        data_batch = list(data_batch)  # in case of data_batch is tuple
        for idx, batch in enumerate(data_batch):
            child_name_str = f'{name_str}.{idx}'
            decollate_op = self.collator.get_decollate_op(batch, child_name_str)
            data_batch[idx] = decollate_op(
                batch, child_name_str, batch_info, is_collate=False)

        data_list = [[] for _ in range(len(data_batch[0]))]
        for idx in range(len(data_list)):
            data_list[idx] = [data[idx] for data in data_batch]

        return data_list

    def __decollate_tuple(self, data_batch, name_str, batch_info):
        return tuple(self.__decollate_list(data_batch, name_str, batch_info))

    def __decollate_dict(self, data_batch, name_str, batch_info):
        batch_size = batch_info['size']
        for key, batch in data_batch.items():
            child_name_str = f'{name_str}.{key}'
            decollate_op = self.collator.get_decollate_op(batch, child_name_str)
            data_batch[key] = decollate_op(
                batch, child_name_str, batch_info, is_collate=False)

        data_list = [{} for _ in range(batch_size)]
        for idx in range(len(data_list)):
            data_list[idx] = {key: data[idx]
                              for key, data in data_batch.items()}

        return data_list


COLLATE_OPERATORS = {
    'unique': CollateOperatorUnique,
    'append': CollateOperatorAppend,
    'cat': CollateOperatorCat,
    'stack': CollateOperatorStack,
    'recursive': CollateOperatorRecusive,
}
