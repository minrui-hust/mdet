from functools import partial
import math
import os

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mdet.core.annotation import Annotation3d
import mdet.model.loss.loss as loss
from mdet.ops.iou3d import nms
from mdet.utils.factory import FI
from mdet.utils.gaussian import draw_gaussian, gaussian_radius

from .base_codec import BaseCodec


@FI.register
class CenterPointCodec(BaseCodec):
    r'''
    Assign heatmap, offset, height, size, cos_theta, sin_theta, positive_indices
    offset is the box center to voxel center,
    size is in log format
    '''

    #  def __init__(self, point_range, grid_size, grid_reso, min_gaussian_radius=1, min_gaussian_overlap=0.5, data_enable=True, anno_enable=True):
    def __init__(self, encode_cfg, decode_cfg, loss_cfg):
        super().__init__(encode_cfg, decode_cfg, loss_cfg)

        # for encode
        # float [x_min, y_min, z_min, x_max, y_max, z_max]
        self.point_range = np.array(
            self.encode_cfg['point_range'], dtype=np.float32)
        # int [reso_x, reso_y, reso_z]
        self.grid_reso = np.array(self.encode_cfg['grid_reso'], dtype=np.int32)
        # float [size_x, size_y, size_z]
        self.grid_size = np.array(
            self.encode_cfg['grid_size'], dtype=np.float32)
        self.min_gaussian_radius = self.encode_cfg['min_gaussian_radius']
        self.min_gaussian_overlap = self.encode_cfg['min_gaussian_overlap']

        # for decode
        # TODO

        # for loss
        self.criteria_heatmap = partial(
            loss.focal_loss, alpha=self.loss_cfg['alpha'], beta=self.loss_cfg['beta'])
        self.criteria_regression = partial(F.l1_loss, reduction='mean')

    def encode_data(self, sample, info):
        # just pcd for now
        sample['input'] = dict(pcd=torch.from_numpy(
            sample['data']['pcd'].points))

    def encode_anno(self, sample, info):
        boxes = sample['anno'].boxes
        types = sample['anno'].types
        type_num = len(sample['meta']['type_name'])

        # offset
        cords_x = np.floor(
            (boxes[:, 0] - self.point_range[0]) / self.grid_size[0])
        cords_y = np.floor(
            (boxes[:, 1] - self.point_range[1]) / self.grid_size[1])
        cords = np.stack((cords_x, cords_y), axis=-1).astype(np.int32)

        grid_offset = self.point_range[:2] + self.grid_size[:2]/2
        center = cords * self.grid_size[:2] + grid_offset
        offset = boxes[:, :2] - center

        # positive_indices
        positive_indices = np.stack(
            (cords_y, cords_x), axis=-1).astype(np.int32)

        positive_heatmap_indices = np.stack(
            (types, cords_y, cords_x), axis=-1).astype(np.int32)

        # height
        height = boxes[:, [2]]

        # size, in log format
        size = np.log(boxes[:, 3:6])

        # heading, in complex number format
        heading = np.stack((np.cos(boxes[:, 6]), np.sin(boxes[:, 6])), axis=-1)

        # heatmap
        heatmap = np.zeros(
            (type_num, self.grid_reso[1], self.grid_reso[0]), dtype=np.float32)
        for i in range(len(boxes)):
            box = boxes[i]
            type = types[i]
            center = cords[i]

            # skip object out of bound
            if not(center[0] >= 0 and center[0] < self.grid_reso[0] and
                    center[1] >= 0 and center[1] < self.grid_reso[1]):
                continue

            l, w = box[3] / self.grid_size[0], box[4] / self.grid_size[1]
            radius = gaussian_radius((l, w), self.min_gaussian_overlap)
            radius = max(self.min_gaussian_radius, int(radius))
            draw_gaussian(heatmap[type], center, radius)

        sample['gt'] = dict(offset=torch.from_numpy(offset),
                            height=torch.from_numpy(height),
                            size=torch.from_numpy(size),
                            heading=torch.from_numpy(heading),
                            heatmap=torch.from_numpy(heatmap),
                            positive_indices=torch.from_numpy(positive_indices),
                            positive_heatmap_indices=torch.from_numpy(
                                positive_heatmap_indices),
                            )

    def decode(self, output, batch):
        r'''
        output --> pred
        '''

        heatmap = torch.sigmoid(output['heatmap'])
        B, C, H, W = heatmap.shape

        score, label = heatmap.max(dim=1)
        score = score.view(B, H*W)
        label = label.view(B, H*W)

        boxes = torch.cat([
            output['offset'],
            output['height'],
            output['size'],
            output['heading'],
        ], dim=1)
        boxes = boxes.permute(0, 2, 3, 1).view(B, H*W, 8)

        # topk_socre: [B, K], topk_indices: [B, K], K = pre_nms_num
        topk_score, topk_indices = score.topk(
            self.decode_cfg['nms_cfg']['pre_num'])

        topk_boxes = boxes.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, 8))  # [B, K, 8]
        topk_label = label.gather(1, topk_indices)  # [B, K]

        # decode boxes into standard format
        topk_boxes = self.decode_box(topk_boxes, topk_indices)  # [B, K, 7]

        # box for nms, xyxyr
        nms_boxes = torch.zeros_like(topk_boxes[..., :5])
        half_w = topk_boxes[..., 3] / 2
        half_h = topk_boxes[..., 4] / 2
        nms_boxes[..., 0] = topk_boxes[..., 0] - half_w
        nms_boxes[..., 1] = topk_boxes[..., 1] - half_h
        nms_boxes[..., 2] = topk_boxes[..., 0] + half_w
        nms_boxes[..., 3] = topk_boxes[..., 1] + half_h
        nms_boxes[..., 4] = -topk_boxes[..., 6]

        # do nms for each sample
        pred_list = []
        for i in range(batch['_info_']['size']):
            keep_indices, valid_num = nms(
                nms_boxes[i],
                topk_score[i],
                self.decode_cfg['nms_cfg']['post_num'],
                self.decode_cfg['nms_cfg']['overlap_thresh'],
            )
            valid_indices = keep_indices[:valid_num]
            det_box = topk_boxes[i][valid_indices]
            det_label = topk_label[i][valid_indices]
            det_score = topk_score[i][valid_indices]
            pred = Annotation3d(
                boxes=det_box.cpu().numpy(), types=det_label.cpu().numpy(), scores=det_score.cpu().numpy())
            pred_list.append(pred)

        return pred_list

    def loss(self, output, batch):
        positive_index = batch['gt']['positive_indices'].long()
        positive_heatmap_index = batch['gt']['positive_heatmap_indices'].long()

        loss_dict = {}
        for head_name, head_prediction in output.items():
            if head_name == 'heatmap':
                heatmap_prediction = self.safe_sigmoid(head_prediction)
                loss = self.criteria_heatmap(
                    heatmap_prediction, batch['gt'][head_name], positive_heatmap_index)
            else:
                # select the positive predictions
                positive_prediction = head_prediction[positive_index[:,
                                                                     0], :, positive_index[:, 1], positive_index[:, 2]]
                loss = self.criteria_regression(
                    positive_prediction, batch['gt'][head_name])
            loss_dict[f'loss_{head_name}'] = loss

        # calc the total loss of different heads
        assert(len(self.loss_cfg['head_weight']) == len(loss_dict))
        loss = 0
        for head_name in self.loss_cfg['head_weight']:
            head_loss = loss_dict[f'loss_{head_name}']
            head_wgt = self.loss_cfg['head_weight'][head_name]
            loss += head_wgt*head_loss

        loss_dict['loss'] = loss

        return loss_dict

    def get_collater(self):
        collator_cfg = dict(
            type='SimpleCollator',
            rules={
                # rules for data
                '.data': dict(type='append'),

                # rules for anno
                '.anno': dict(type='append'),

                # rules for input
                '.input.pcd': dict(type='append'),

                # rules for gt
                '.gt.offset': dict(type='cat'),
                '.gt.height': dict(type='cat'),
                '.gt.size': dict(type='cat'),
                '.gt.heading': dict(type='cat'),
                '.gt.heatmap': dict(type='stack'),
                '.gt.positive_indices': dict(
                    type='cat',
                    dim=0,
                    pad_cfg=dict(pad=(1, 0)),
                    inc_func=lambda x: torch.tensor(
                        [1, 0, 0], dtype=torch.int32),
                ),
                '.gt.positive_heatmap_indices': dict(
                    type='cat',
                    dim=0,
                    pad_cfg=dict(pad=(1, 0)),
                    inc_func=lambda x: torch.tensor(
                        [1, 0, 0, 0], dtype=torch.int32),
                ),


                # rules for output, would be very simillar to gt
                '.output.offset': dict(type='stack'),
                '.output.height': dict(type='stack'),
                '.output.size': dict(type='stack'),
                '.output.heading': dict(type='stack'),
                '.output.heatmap': dict(type='stack'),

                # rules for meta
                '.meta': dict(type='append'),
            },
        )

        return FI.create(collator_cfg)

    def plot(self, sample, show_pcd=True, show_heatmap_gt=False, show_heatmap_pred=True, show_box=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        if show_pcd:
            pcd = sample['input']['pcd']
            ax.scatter(pcd[:, 0], pcd[:, 1], c='black',
                       marker='.', alpha=0.7, s=1)

        if show_heatmap_gt:
            heatmap = sample['gt']['heatmap']
            heatmap = torch.max(heatmap, dim=0)[0]
            heatmap = torch.sigmoid(heatmap)
            x = self.point_range[0] + self.grid_size[0] * \
                np.arange(self.grid_reso[0])
            y = self.point_range[1] + self.grid_size[1] * \
                np.arange(self.grid_reso[1])
            ax.pcolormesh(x, y, heatmap, cmap='hot', alpha=0.6)

        if show_heatmap_pred:
            heatmap = sample['output']['heatmap']
            heatmap = torch.max(heatmap, dim=0)[0]
            heatmap = torch.sigmoid(heatmap)
            x = self.point_range[0] + self.grid_size[0] * \
                np.arange(self.grid_reso[0])
            y = self.point_range[1] + self.grid_size[1] * \
                np.arange(self.grid_reso[1])
            ax.pcolormesh(x, y, heatmap, cmap='hot', alpha=0.6)

        if show_box:
            positive_indices = sample['gt']['positive_indices']
            cord = torch.stack(
                [positive_indices[:, 1], positive_indices[:, 0]], dim=-1)
            grid_offset = self.point_range[:2] + self.grid_size[:2]/2
            center = cord * self.grid_size[:2] + grid_offset
            offset = sample['gt']['offset']
            size = sample['gt']['size']
            cos_rot = sample['gt']['heading'][:, 0]
            sin_rot = sample['gt']['heading'][:, 1]
            rot = torch.atan2(sin_rot, cos_rot) * 180 / math.pi
            l = torch.exp(size[:, 0])
            w = torch.exp(size[:, 1])
            dx = (l*cos_rot - w*sin_rot)/2
            dy = (l*sin_rot + w*cos_rot)/2
            pos = center + offset - torch.stack([dx, dy], dim=-1)
            for i in range(len(pos)):
                rect = Rectangle(
                    (pos[i, 0].item(), pos[i, 1].item()), l[i].item(), w[i].item(), angle=rot[i], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        plt.axis('equal')
        plt.show()

    def safe_sigmoid(self, x):
        return torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)

    def decode_box(self, boxes, cords):
        grid_offset = self.point_range[:2] + self.grid_size[:2]/2
        grid_offset = torch.from_numpy(grid_offset).to(boxes.device)
        grid_size = torch.from_numpy(self.grid_size[:2]).to(boxes.device)
        grid_reso = torch.from_numpy(self.grid_reso[:2]).to(boxes.device)

        # if is B x N, convert to B x N x 2
        if cords.dim() == 2:
            cords_y = cords//grid_reso[0]
            cords_x = cords % grid_reso[0]
            cords = torch.stack([cords_x, cords_y], dim=-1)

        grid_center = cords * grid_size + grid_offset

        decoded_box_xy = boxes[..., :2] + grid_center

        decoded_box_z = boxes[..., [2]]

        decoded_box_size = torch.exp(boxes[..., 3:6])

        decoded_box_rot = torch.atan2(boxes[..., [7]], boxes[..., [6]])

        decoded_box = torch.cat([
            decoded_box_xy,
            decoded_box_z,
            decoded_box_size,
            decoded_box_rot
        ], dim=-1)

        return decoded_box
