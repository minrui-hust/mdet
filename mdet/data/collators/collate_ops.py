import numpy as np
from mdet.utils.misc import is_seq_of


class CollateOperator(object):
    def __init__(self, collator):
        super().__init__()
        self.collator = collator

    def __call__(self, sample_list, name_str):
        raise NotImplementedError


class CollateOperatorUnique(CollateOperator):
    def __init__(self, collator):
        super().__init__(collator)

    def __call__(self, data_list, name_str):
        if not data_list:
            return None
        return data_list[0]


class CollateOperatorAppend(CollateOperator):
    def __init__(self, collator):
        super().__init__(collator)

    def __call__(self, data_list, name_str):
        if not data_list:
            return None

        return data_list


class CollateOperatorCat(CollateOperator):
    def __init__(self, collator, dim=0, pad=None, inc_func=None):
        super().__init__(collator)
        self.dim = dim
        self.inc_func = inc_func
        self.pad = pad

    def __call__(self, data_list, name_str):
        if not data_list:
            return None

        new_data_list = []
        acc_list = [0]
        for data in data_list:
            inc = 0 if self.inc_func is None else self.inc_func(data)

            if self.pad is not None:
                data = np.pad(data, **self.pad)

            new_data_list.append(data + acc_list[-1])

            acc_list.append(acc_list[-1]+inc)

        return np.concatenate(new_data_list, axis=self.dim)


class CollateOperatorStack(CollateOperator):
    def __init__(self, collator, dim=0, inc_func=None):
        super().__init__(collator)
        self.dim = dim
        self.inc_func = inc_func

    def __call__(self, data_list, name_str):
        if not data_list:
            return None

        if self.inc_func is None:
            return np.stack(data_list, axis=self.dim)
        else:
            inc = 0
            new_data_list = []
            for data in data_list:
                new_data_list.append(data+inc)
                inc += self.inc_func(data)
            return np.stack(new_data_list, axis=self.dim)


class CollateOperatorRecusive(CollateOperator):
    def __init__(self, collator):
        super().__init__(collator)

    def __call__(self, data_list, name_str):
        if not data_list:
            return None
        if is_seq_of(data_list, list) or is_seq_of(data_list, tuple):
            return self.__collate_sequence(data_list, name_str)
        elif is_seq_of(data_list, dict):
            return self.__collate_dict(data_list, name_str)
        else:
            raise NotImplementedError

    def __collate_sequence(self, data_list, name_str):
        # init the list
        batch = [[] for _ in range(len(data_list[0]))]

        # collect samples
        for sample in data_list:
            for idx, value in enumerate(sample):
                batch[idx].append(value)

        # do collate
        for idx, value_list in enumerate(batch):
            child_name_str = f'{name_str}.{idx}'
            collate_op = self.collator.get_operator(value_list, child_name_str)
            batch[idx] = collate_op(value_list, child_name_str)

        return batch

    def __collate_dict(self, data_list, name_str):
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
            collate_op = self.collator.get_operator(value_list, child_name_str)
            batch[key] = collate_op(value_list, child_name_str)

        return batch


COLLATE_OPERATORS = {
    'unique': CollateOperatorUnique,
    'append': CollateOperatorAppend,
    'cat': CollateOperatorCat,
    'stack': CollateOperatorStack,
    'recursive': CollateOperatorRecusive,
}
