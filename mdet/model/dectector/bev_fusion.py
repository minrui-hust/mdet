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
    def __init__(self, voxelization, backbone3d, backbone2d=None):
        super().__init__()

        self.voxelization = FI.create(voxelization)
        self.backbone3d = FI.create(backbone3d)
        self.backbone2d = FI.create(backbone2d)

    def forward_train(self, points_list):
        out = self.voxelization(points_list)
        out = self.backbone3d(out)
        if self.backbone2d is not None:
            out = self.backbone2d(out)[-1]
        return out


@FI.register
class ImgEncoder(BaseModule):
    def __init__(self, backbone, neck):
        super().__init__()

        self.backbone = FI.create(backbone)
        self.neck = FI.create(neck)

    def forward_train(self, img):
        r'''
        Args:
            img: shape (B, C, F, H, W)
        '''
        out = self.backbone(img.view(-1, *img.shape[2:]))
        out = self.neck(out)
        return out


@FI.register
class BevTransformerFusion(BaseModule):
    def __init__(self, transformer, query_info, key_info):
        super().__init__()

        self.decoder = FI.create(transformer)

        self.query_W = query_info['grid_reso'][0]
        self.query_H = query_info['grid_reso'][1]
        self.query_Z = query_info['grid_reso'][2]

        self.key_W0 = key_info['spatial_shapes'][0][0]
        self.key_H0 = key_info['spatial_shapes'][0][1]
        self.key_level = len(key_info['spatial_shapes'])

        # query positional_encoding
        query_pe = SinPositionalEncoding2D(
            query_info['grid_reso'][1], query_info['grid_reso'][0], query_info['d_model'])
        self.register_buffer('query_pe', query_pe)

        # query ref point3d, need cam calibrations to convert to ref point 2d
        query_ref_point3d = self.query_ref_3d(
            query_info['grid_range'], query_info['grid_reso'], query_info['grid_size'])
        self.register_buffer('query_ref_point3d', query_ref_point3d)

        # key positional_encoding
        for i, (w, h) in enumerate(key_info['spatial_shapes']):
            key_pe = SinPositionalEncoding2D(h, w, key_info['d_model'])
            self.register_buffer(f'key_pe_{i}', key_pe)

        # key spatial shapes
        key_spatial_shapes = torch.empty(self.key_level, 2, dtype=torch.int64)
        key_level_index = [0]
        for i, (w, h) in enumerate(key_info['spatial_shapes']):
            key_spatial_shapes[i, 0] = h
            key_spatial_shapes[i, 1] = w
            key_level_index.append(key_level_index[-1]+h*w)
        self.register_buffer('key_spatial_shapes', key_spatial_shapes)

        key_level_index = torch.tensor(key_level_index[:-1], dtype=torch.int64)
        self.register_buffer('key_level_index', key_level_index)

    def forward_train(self, query, key_list, cam_calibs):
        r'''
        Args:
            query: input query, shape (B, F, H, W)
            key_list: input keys of multi level, each level is in shape (B*C, F, H, W), C is camera number
            cam_calibs: camera calibrations, in shape (B, C, 3, 4), [KR|Kt]
        '''
        B, H, W, C, L, Z = query.shape[0], query.shape[2], query.shape[3], cam_calibs.shape[1], self.key_level, self.query_Z

        # add positional_encoding and reshape
        query = (query + self.query_pe).permute(0, 2, 3, 1).reshape(B, H*W, -1)

        key_list_reshaped = []
        for i in range(len(key_list)):
            h, w = key_list[i].shape[2:4]
            pe = self.__getattr__(f'key_pe_{i}')
            key_list_reshaped.append(
                (key_list[i] + pe).permute(0, 2, 3, 1).reshape(B*C, h*w, -1))
        key = torch.concat(key_list_reshaped, dim=1)

        # calc ref point and mask based on cam_calibs
        ref_point, ref_mask, ref_scale = self.ref_3d_to_2d(self.query_ref_point3d, cam_calibs)
        ref_point = ref_point.view(B*C, Z*H*W, 1, 2).expand(-1, -1, L, -1)
        ref_mask = ref_mask.view(B*C, Z*H*W)
        ref_scale = ref_scale.view(B, H*W)

        # do fusion, shape (B, H*W, F)
        out = self.decoder(query, key,
                           spatial_shapes=self.key_spatial_shapes,
                           level_index=self.key_level_index,
                           ref_point=ref_point,
                           ref_mask=ref_mask,
                           ref_scale=ref_scale)

        out = out.permute(0, 2, 1).reshape(B, -1, H, W)

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

    def ref_3d_to_2d(self, point3d, calibs, eps=1e-6):
        r'''
        Args:
            point3d: in shape (Z, H, W, 3)
            calibs: in shape (B, C, 3, 4), [KR | Kt]
        Return:
            ref_point: in shape (B, C, Z, H, W, 2)
            ref_mask: in shape (B, C, Z, H, W)
            ref_scale: in shape (B, H, W)
        '''
        B, C, Z, H, W = *calibs.shape[:2], *point3d.shape[:3]

        KR = calibs[...,:3,:3].view(B, C, 1, 1, 3, 3)
        Kt = calibs[...,:3, 3].view(B, C, 1, 1, 1, 3)

        point3d = point3d.view(1, 1, Z, H, W, 3)

        # (B, C, Z, H, W, 3)
        img_point = torch.matmul(point3d, KR.transpose(4,5)) + Kt

        # (B, C, Z, H, W, 2)
        ref_point = img_point[..., :2]/(img_point[..., [2]] + eps)
        ref_point[..., 0] /= self.key_W0
        ref_point[..., 1] /= self.key_H0

        # (B, C, Z, H, W), true to mask, false keep
        ref_mask = torch.logical_or(img_point[..., 2] < 0.5,
                                    torch.logical_or(torch.any(ref_point < 0, dim=-1), torch.any(ref_point > 1, dim=-1)))

        # (B, H, W)
        ref_rank = torch.sum(torch.logical_not(
            torch.all(ref_mask, dim=2)), dim=1)
        ref_scale = torch.div(1, ref_rank)

        return ref_point,  ref_mask, ref_scale


@FI.register
class BevDeformCrossAttn(BaseModule):
    def __init__(self, d_model, n_levels, n_heads, n_points, n_cams, grid_reso):
        super().__init__()

        self.n_cams = n_cams
        self.grid_reso = grid_reso

        self.deform_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def forward_train(self, x, y, spatial_shapes, level_index, ref_point, ref_mask, ref_scale):
        r'''
        Args:
            x: the query, in shape (B, H*W, F)
            y: the key, in shape (B*C, h0*w0+h1*w1+..., f)
            spatial_shapes: in shape (L, 2)
            level_index: in shape (L,)
            ref_point: in shape (B*C, Z*H*W, L, 2)
            ref_mask: in shape (B*C, Z*H*W)
            ref_scale: in shape (B, H*W)

        Return:
            out: output feature after fusion, in shape (B, H*W, F)
        '''
        B, C, W, H, Z = x.shape[0], self.n_cams, *self.grid_reso

        x = x.view(B, 1, 1, H*W, -1).expand(-1, C,
                                            Z, -1, -1).reshape(B*C, Z*H*W, -1)

        # shape (B*C, Z*H*W, F)
        x = self.deform_attn(x, y,
                             reference_points=ref_point,
                             input_spatial_shapes=spatial_shapes,
                             input_level_start_index=level_index)

        # mask invalid value
        x.masked_fill_(ref_mask.unsqueeze(-1), 0)

        # sum and scale
        x = torch.sum(x.view(B, C*Z, H*W, -1), dim=1) * ref_scale.unsqueeze(-1)

        return x


@FI.register
class BevDeformSelfAttn(BaseModule):
    def __init__(self, d_model, n_heads, n_points, grid_reso):
        super().__init__()

        self.deform_attn = MSDeformAttn(
            d_model, n_levels=1, n_heads=n_heads, n_points=n_points)

        ref_point, spatial_shape, level_index = self.spatial_info(grid_reso)
        self.register_buffer('ref_point', ref_point)
        self.register_buffer('spatial_shape', spatial_shape)
        self.register_buffer('level_index', level_index)

    def forward_train(self, x):
        r'''
        Args:
            x: in shape (B, H*W, F)
        '''
        ref_point = self.ref_point.view(
            1, -1, 1, 2).expand(x.shape[0], -1, -1, -1)

        out = self.deform_attn(x, x,
                               reference_points=ref_point,
                               input_spatial_shapes=self.spatial_shape,
                               input_level_start_index=self.level_index)
        return out

    def spatial_info(self, grid_reso):
        grid_reso = np.array([grid_reso[1], grid_reso[0]], dtype=np.int32)
        grid_size = np.array(
            [1/grid_reso[1], 1/grid_reso[0]], dtype=np.float32)

        grid = np.mgrid[:grid_reso[0], :grid_reso[1]
                        ].transpose(1, 2, 0).reshape(-1, 2).astype(np.float32)

        # (H*W, 2)
        ref_point = grid * grid_size + 0.5*grid_size

        # (1, 2)
        spatial_shape = grid_reso[np.newaxis, :].astype(np.int64)

        # (1,)
        level_index = np.array([0], dtype=np.int64)

        return torch.from_numpy(ref_point), torch.from_numpy(spatial_shape), torch.from_numpy(level_index)


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
        fusion_out = self.fusion(pcd_out, img_out, batch['input']['calibs'])
        bb2d_out = self.backbone2d(fusion_out)
        neck_out = self.neck(bb2d_out)
        head_out = self.head(neck_out)
        return head_out
