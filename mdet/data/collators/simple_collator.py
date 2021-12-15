from mdet.data.sample import Sample
from mdet.utils.misc import is_seq_of
from mdet.data.collators.collate_ops import COLLATE_OPERATORS
from mdet.utils.factory import FI


@FI.register
class SimpleCollator(object):
    def __init__(self, rules={}):
        super().__init__()

        self.rules = rules

    def __call__(self, sample_list):
        r'''
        '''

        collate_op = self.get_operator(sample_list, '')
        batch = collate_op(sample_list, '')

        return Sample(**batch)

    def get_operator(self, value_list, name_str):
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
