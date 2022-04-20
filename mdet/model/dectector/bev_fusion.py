import torch
import torch.nn as nn
import torch.nn.functional as ff

import numpy as np
from mdet.utils import rigid

from mai.utils import FI
from mai.model import BaseModule
from mai.model.bricks.attention import MSDeformAttn
from mai.model.utils.positional_encoding import SinPositionalEncoding2D


@FI.register
class PcdEncoder(BaseModule):
    def __init__(self, voxelization, backbone3d):
        super().__init__()

        self.voxelization = FI.create(voxelization)
        self.backbone3d = FI.create(backbone3d)

    def forward_train(self, points_list):
        out = self.voxelization(points_list)
        out = self.backbone3d(out)
        return out


@FI.register
class ImgEncoder(BaseModule):
    def __init__(self, backbone, neck=None):
        super().__init__()

        self.backbone = FI.create(backbone)
        self.neck = FI.create(neck)

    def forward_train(self, img):
        out = self.backbone(img)
        if self.neck is not None:
            out = self.neck(out)
        return out


@FI.register
class BevTransformerFusion(BaseModule):
    def __init__(self, transformer, query_info, key_info):
        super().__init__()

        self.decoder = FI.create(transformer)

        self.key_H0 = key_info[0]['H']
        self.key_W0 = key_info[0]['W']
        self.key_level = len(key_info)

        # query positional_encoding
        query_pe = SinPositionalEncoding2D(
            query_info['grid_reso'][1], query_info['grid_reso'][0], query_info['d_model'])
        self.register_buffer('query_pe', query_pe)

        # query ref point3d, need cam calibrations to convert to ref point 2d
        query_ref_point3d = self.query_ref_3d(
            query_info['grid_range'], query_info['grid_reso'], query_info['grid_size'])
        self.register_buffer('query_ref_point3d', query_ref_point3d)

        # key positional_encoding
        for i, info in enumerate(key_info):
            key_pe = SinPositionalEncoding2D(
                info['H'], info['W'], info['d_model'])
            self.register_buffer(f'key_pe_{i}', key_pe)

        # key spatial shapes
        key_spatial_shapes = torch.empty(len(key_info), 2, dtype=torch.int32)
        key_level_index = [0]
        for i, info in enumerate(key_info):
            key_spatial_shapes[i, 0] = info['H']
            key_spatial_shapes[i, 1] = info['W']
            key_level_index.append(key_level_index[-1]+info['H']*info['W'])
        self.register_buffer('key_spatial_shapes', key_spatial_shapes)

        key_level_index = torch.tensor(key_level_index[:-1], dtype=torch.int32)
        self.register_buffer('key_level_index,', key_level_index)

    def forward_train(self, query, key_list, cam_calibs):
        r'''
        Args:
            query: input query, shape (B, F, H, W)
            key_list: input keys of multi level, each level is in shape (B, C, F, H, W), C is camera number
            cam_calibs: camera calibrations, tuple of (extrinsics, intrinsics), extrinsics in shape (C, 4, 4)
                intrinsics in shape (C, 3, 3)
        '''
        B, H, W, C, L = query.shape[0], query.shape[2], query.shape[3], key_list[0].shape[1], self.key_level

        # add positional_encoding and reshape
        query = (query.permute(0, 2, 3, 1) + self.query_pe).reshape(B, H*W, -1)

        key_list_reshaped = []
        for i in range(len(key_list)):
            h, w = key_list[i].shape[2:4]
            key_list_reshaped.append((key_list[i].permute(
                0, 1, 3, 4, 2) + self.__getattr__(f'key_pe_{i}')).reshape(B*C, h*w, -1))
        key = torch.concat(key_list_reshaped, dim=1)

        # calc ref point and mask based on cam_calibs
        ref_points, ref_mask = self.ref_3d_to_2d(
            self.query_ref_point3d, cam_calibs[0], cam_calibs[1])
        ref_points = ref_points.view(
            1, C, -1, 1, 2).expand(B, -1, -1, L, -1).reshape(B*C, -1, L, 2)
        ref_mask = ref_mask.unsqueeze(0).expand(B, -1, -1).reshape(B*C, -1)

        # do fusion
        out = self.decoder(query, key,
                           key_spatial_shapes=self.key_spatial_shapes,
                           key_level_index=self.key_level_index,
                           ref_points=ref_points,
                           ref_mask=ref_mask)

        return out

    def query_ref_3d(self, grid_range, grid_reso, grid_size):
        r'''
        Return: the 3d query ref points in shape (Z, H, W, 3)
        '''

        grid_range = np.array(
            [grid_range[2], grid_range[1], grid_range[0]], dtype=np.float32)
        grid_size = np.array(
            [grid_size[2], grid_size[1], grid_size[0]], dtype=np.float32)
        grid_reso = np.array(
            [grid_reso[2], grid_reso[1], grid_reso[0]], dtype=np.int32)

        grid = np.mgrid[:grid_reso[0], :grid_reso[1],
                        :grid_reso[2]].transpose(1, 2, 3, 0).astype(np.float32)

        # shape (Z, H, W, 3)
        point3d = grid * grid_size - grid_range[:3] + 0.5*grid_size

        return torch.from_numpy(point3d)

    def ref_3d_to_2d(self, point3d, extrinsics, intrinsics):
        r'''
        Args:
            point3d: in shape (Z, H, W, 3)
            extrinsics: in shape (C, 4, 4)
            intrinsics: inshape (C, 3, 3)
        Return:
            ref_point: in shape (C, Z*H*W, 2)
            ref_mask: in shape (C, Z*H*W)
        '''

        R = extrinsics[:, :3, :3]  # (C, 3, 3)
        t = extrinsics[:, :3, 3].unsqueeze(1)  # (C, 1, 3)

        point3d = point3d.view(1, -1, 3)   # (1, Z*H*W, 3)

        # (C, Z*H*W, 3)
        cam_point = torch.matmul(point3d, R.transpose(1, 2)) + t

        # (C, Z*H*W, 3)
        img_point = torch.matmul(cam_point, intrinsics.transpose(1, 2))

        ref_mask = img_point[:, :, 2] > 0.5

        ref_point = img_point[:, :, :2]/(img_point[:, :, [2]], 1e-6)
        ref_point[:, :, 0] = ref_point[:, :, 0] / self.key_W0
        ref_point[:, :, 1] = ref_point[:, :, 1] / self.key_H0

        return ref_point,  ref_mask


@FI.register
class BevDeformAttn(BaseModule):
    def __init__(self, d_model, n_levels, n_heads, n_points, n_cams, grid_reso):
        super().__init__()

        self.n_cams = n_cams
        self.grid_reso = grid_reso

        self.deform_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def forward_train(self, x, y, spatial_shapes, level_index, ref_point, ref_mask):
        r'''
        Args:
            x: the query, in shape (B, H*W, F)
            y: the key, in shape (B*C, h0*w0+h1*w1+..., f)
            spatial_shapes: in shape (L, 2)
            level_index: in shape (L,)
            ref_point: in shape (B*C, Z*H*W, L, 2)
            ref_mask: in shape (B*C, Z*H*W)

        Return:
            out: output feature after fusion
        '''
        B = x.shape[0]
        C = self.n_cams
        W, H, Z = self.grid_reso

        x = x.view(B, 1, 1, H*W, -1).expand(-1, C,
                                            Z, -1, -1).reshape(B*C, Z*H*W, -1)

        # shape (B*C, Z*H*W, F)
        x = self.deform_attn(x, y,
                             reference_points=ref_point,
                             input_spatial_shapes=spatial_shapes,
                             input_level_start_index=level_index)

        # mask unused value
        x.masked_fill_(ref_mask.unsqueeze(-1))

        x = torch.sum(x.view(B, C*Z, H*W, -1), dim=1)  # (B, HW, F)

        return x


@FI.register
class BevFusionNet(BaseModule):
    def __init__(self, pcd_encoder, img_encoder, fusion, backbone2d, neck, head):
        super().__init__()

        self.pcd_encoder = FI.create(pcd_encoder)
        self.img_encoder = FI.create(img_encoder)
        self.fusion = FI.create(fusion)
        self.backbone2d = FI.create(backbone2d)
        self.neck = FI.create(neck)
        self.head = FI.create(head)

    def forward_train(self, batch):
        pcd_out = self.pcd_encoder(batch['input']['points'])
        img_out = self.img_encoder(batch['input']['images'])
        fusion_out = self.fusion(pcd_out, img_out)
        bb2d_out = self.backbone2d(fusion_out)
        neck_out = self.neck(bb2d_out)
        head_out = self.head(neck_out)
        return head_out
