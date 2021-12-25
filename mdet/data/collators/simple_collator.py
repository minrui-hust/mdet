from mdet.data.sample import Sample
from mdet.utils.misc import is_seq_of, is_list_of
from mdet.data.collators.collate_ops import COLLATE_OPERATORS
from mdet.utils.factory import FI


@FI.register
class SimpleCollator(object):
    def __init__(self, rules={}):
        super().__init__()
        self.rules = rules

    def __call__(self, sample_list_or_batch, is_collate=True):
        if is_collate:
            return self.collate(sample_list_or_batch)
        else:
            return self.decollate(sample_list_or_batch)

    def collate(self, sample_list):
        if not sample_list:
            return None

        batch_info = dict(size=len(sample_list))
        collate_op = self.get_collate_op(sample_list, '')
        sample_batch = collate_op(sample_list, '', batch_info)
        sample_batch['_info_'] = batch_info

        return Sample(**sample_batch)

    def decollate(self, sample_batch):
        if not sample_batch:
            return None

        batch_info = sample_batch.pop('_info_')
        decollate_op = self.get_decollate_op(sample_batch, '')
        sample_list = decollate_op(
            sample_batch, '', batch_info, is_collate=False)

        if sample_list:
            return [Sample(**sample) for sample in sample_list]
        else:
            return None

    def get_collate_op(self, value_list, name_str):
        if name_str == '':
            return COLLATE_OPERATORS['recursive'](self)
        else:
            for rule in self.rules.keys():
                if self.match(name_str, rule):
                    args = self.rules[rule].copy()
                    op_type = args.pop('type')
                    return COLLATE_OPERATORS[op_type](self, **args)

        # no rules found, get default
        if is_seq_of(value_list, dict) or is_seq_of(value_list, list) or is_seq_of(value_list, tuple):
            return COLLATE_OPERATORS['recursive'](self)
        else:
            return COLLATE_OPERATORS['append'](self)

    def get_decollate_op(self, value_batch, name_str):
        if name_str == '':
            return COLLATE_OPERATORS['recursive'](self)
        else:
            for rule in self.rules.keys():
                if self.match(name_str, rule):
                    args = self.rules[rule].copy()
                    op_type = args.pop('type')
                    return COLLATE_OPERATORS[op_type](self, **args)

        # no rules found, get default
        if isinstance(value_batch, dict) or isinstance(value_batch, tuple) or is_list_of(value_batch, list):
            return COLLATE_OPERATORS['recursive'](self)
        else:
            return COLLATE_OPERATORS['append'](self)

    def match(self, name_str, pattern_str):
        r'''
        a.b.c match a.b.c, a.*.c, a.b.* ...
        '''

        name_list = name_str.split('.')
        pattern_list = pattern_str.split('.')

        if len(name_list) != len(pattern_list):
            return False

        for name, pattern in zip(name_list, pattern_list):
            if name != pattern and pattern != '*':
                return False

        return True
