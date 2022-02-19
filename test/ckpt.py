import torch

#  ckpt = torch.load('log/centerpoint_pp_waymo_3cls_small_range_gtaug/version_0/epoch=35-step=355715.ckpt')
ckpt = torch.load('log/centerpoint_pp_waymo_3cls_small_range_xyz/version_0/epoch=35-step=711395.ckpt')
print(ckpt['state_dict'])
