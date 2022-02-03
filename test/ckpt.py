import torch

#  ckpt = torch.load('log/centerpoint_vn_waymoD5_3cls_gtaug/version_1/epoch=35-step=71171.ckpt')
#  ckpt = torch.load('log/centerpoint_vn_waymo_3cls_baseline/version_0/epoch=35-step=355715.ckpt')
ckpt = torch.load('log/centerpoint_vn_waymoD5_3cls_gtaug/version_1/epoch=12-step=25700.ckpt')
print(ckpt['state_dict'])
