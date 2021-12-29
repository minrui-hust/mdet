import torch
import torch.nn as nn


def construct_mask(actual_num, max_num, inverse=False):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Last dimension represent actual number of points in each voxel.
        max_num (int): Max number, which indicate the size of last dimension

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """

    step_num_shape = [1]*actual_num.dim()
    step_num_shape.append(-1)

    # shape: [1,1,...,max_num]
    step_num = torch.arange(max_num, dtype=actual_num.dtype,
                            device=actual_num.device).view(step_num_shape)

    # shape: [1,1,...,1]
    actual_num = actual_num.unsqueeze(-1)

    if inverse:
        return actual_num <= step_num
    else:
        return actual_num > step_num


def build_norm(norm_cfg, shape):
    cfg = norm_cfg.copy()
    type_str = cfg.pop('type')
    return nn.__dict__[type_str](shape, **cfg)
