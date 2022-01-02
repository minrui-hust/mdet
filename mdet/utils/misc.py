from collections import abc
import torch


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def is_nan_or_inf(x, name='tensor'):
    if isinstance(x, list) or isinstance(x, tuple):
        for i, element in enumerate(x):
            is_nan_or_inf(element, f'{name}.{i}')
    elif isinstance(x, dict):
        for k, v in x.items():
            is_nan_or_inf(v, f'{name}.{k}')
    elif torch.is_tensor(x):
        if x.isnan().any():
            print(f'{name} has nan')
            return True
        if x.isinf().any():
            if x.isneginf().any():
                print(f'{name} has -inf')
                return True
            else:
                print(f'{name} has +inf')
                return True
    return False
