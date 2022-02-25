from functools import partial
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from mdet.core.annotation import Annotation3d
import mdet.model.loss.loss as loss
from mdet.ops.iou3d import nms_bev, iou_bev
from mdet.utils.factory import FI

from .base_codec import BaseCodec


@FI.register
class CenterPointCodec(BaseCodec):
    r'''
    Assign heatmap, offset, height, size, cos_theta, sin_theta, positive_indices
    offset is the box center to voxel center,
    size is in log format
    '''

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

        self.heatmap_encoder = FI.create(self.encode_cfg['heatmap_encoder'])

        # many type may map to same label
        self.type_to_label = {}
        self.label_to_type = {}
        for label, type_list in encode_cfg['labels'].items():
            for type in type_list:
                self.type_to_label[type] = label
                # this may loss information, only record last type
                self.label_to_type[label] = type

        # for decode
        self.valid_thresh = self.decode_cfg.get('valid_thresh', 0.1)
        self.iou_rectification = self.decode_cfg.get(
            'iou_rectification', False)
        self.iou_rectification_gamma = self.decode_cfg.get(
            'iou_rectification_gamma', 2)

        # for loss
        self.criteria_heatmap = partial(
            loss.focal_loss, alpha=self.loss_cfg['alpha'], beta=self.loss_cfg['beta'])
        self.criteria_regression = partial(loss.regression_loss, eps=1e-4)
        self.free_heading_label = self.loss_cfg.get('free_heading_label', None)
        self.normlize_rot = self.loss_cfg.get('normlize_rot', False)

    def encode_data(self, sample, info):
        # just pcd for now
        sample['input'] = dict(points=torch.from_numpy(
            sample['data']['pcd'].points))

    def encode_anno(self, sample, info):
        boxes = sample['anno'].boxes
        types = sample['anno'].types
        label_num = len(self.label_to_type)
        labels = np.array([self.type_to_label[type]
                          for type in types], dtype=np.int32)

        # offset
        cords_x = np.floor(
            (boxes[:, 0] - self.point_range[0]) / self.grid_size[0])
        cords_y = np.floor(
            (boxes[:, 1] - self.point_range[1]) / self.grid_size[1])
        cords = np.stack((cords_x, cords_y), axis=-1).astype(np.float32)

        grid_offset = self.point_range[:2] + self.grid_size[:2]/2
        center = cords * self.grid_size[:2] + grid_offset
        offset = boxes[:, :2] - center

        # positive_indices
        positive_indices = np.stack(
            (cords_y, cords_x), axis=-1).astype(np.int32)

        positive_heatmap_indices = np.stack(
            (labels, cords_y, cords_x), axis=-1).astype(np.int32)

        # height
        height = boxes[:, [2]]

        # size, in log format
        size = np.log(boxes[:, 3:6]*2)

        # heading, in complex number format
        heading = boxes[:, 6:]

        # heatmap
        heatmap = np.zeros(
            (label_num, self.grid_reso[1], self.grid_reso[0]), dtype=np.float32)
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]
            center = cords[i]

            # skip object out of bound
            if not(center[0] >= 0 and center[0] < self.grid_reso[0] and
                    center[1] >= 0 and center[1] < self.grid_reso[1]):
                raise AssertionError(f'center: {center}, box: {boxes[i]}')

            self.heatmap_encoder(heatmap[label], box, center)

        sample['gt'] = dict(offset=torch.from_numpy(offset),
                            height=torch.from_numpy(height),
                            size=torch.from_numpy(size),
                            heading=torch.from_numpy(heading),
                            heatmap=torch.from_numpy(heatmap),
                            positive_indices=torch.from_numpy(
                                positive_indices),
                            positive_heatmap_indices=torch.from_numpy(
                                positive_heatmap_indices),
                            )

    def decode_eval(self, output, batch=None):
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
            self.decode_cfg['nms_cfg']['pre_num'], sorted=not self.iou_rectification)

        topk_boxes = boxes.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[2]))  # [B, K, 8]
        topk_boxes = self.decode_box(topk_boxes, topk_indices)  # [B, K, 8]

        topk_label = label.gather(1, topk_indices)  # [B, K]

        if self.iou_rectification:
            # get predicted iou
            iou = output['iou'].view(B, H*W)
            topk_iou = iou.gather(1, topk_indices)
            topk_iou = (topk_iou + 1) * 0.5  # range [-1,1] to [0,1]

            # rectify score
            topk_score = topk_score * \
                torch.pow(topk_iou, self.iou_rectification_gamma)

            # sort by rectified score
            topk_score, rectify_indices = torch.sort(
                topk_score, dim=-1, descending=True)

            # re-order
            topk_boxes = topk_boxes.gather(
                1, rectify_indices.unsqueeze(-1).expand_as(topk_boxes))
            topk_label = topk_label.gather(1, rectify_indices)
            topk_score = topk_score.gather(1, rectify_indices)

        # box for nms (bird eye view)
        nms_boxes = topk_boxes[..., [0, 1, 3, 4, 6, 7]]

        # do nms for each sample
        pred_list = []
        batch_size = 1 if batch is None else batch['_info_']['size']
        for i in range(batch_size):
            keep_indices, valid_num = nms_bev(
                nms_boxes[i],
                topk_score[i],
                self.decode_cfg['nms_cfg']['overlap_thresh'],
                self.decode_cfg['nms_cfg']['post_num'],
            )
            keep_indices = keep_indices.long()
            valid_indices = keep_indices[:valid_num]
            det_box = topk_boxes[i][valid_indices].cpu().numpy()
            det_label = topk_label[i][valid_indices].cpu().numpy()
            det_score = topk_score[i][valid_indices].cpu().numpy()
            det_type = np.array([self.label_to_type[label]
                                for label in det_label], dtype=np.int32)
            mask = det_score > self.valid_thresh
            pred = Annotation3d(
                boxes=det_box[mask], types=det_type[mask], scores=det_score[mask])
            pred_list.append(pred)

        return pred_list

    def decode_infer(self, output, batch=None):
        r'''
        output --> pred
        '''

        heatmap = torch.sigmoid(output['heatmap'])
        B, C, H, W = heatmap.shape
        assert B == 1

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
            self.decode_cfg['nms_cfg']['pre_num'], sorted=not self.iou_rectification)

        topk_boxes = boxes.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[2]))  # [B, K, 8]
        topk_label = label.gather(1, topk_indices)  # [B, K]

        if self.iou_rectification:
            # get predicted iou
            iou = output['iou'].permute(0, 2, 3, 1).view(B, H*W, -1)
            topk_iou = iou.gather(
                1, topk_indices.unsqueeze(-1).expand(-1, -1, iou.shape[2]))
            topk_iou = topk_iou.gather(1, topk_label)

            # rectify score
            topk_score *= torch.pow(topk_iou, self.iou_rectification_gamma)

            # sort by rectified score
            topk_score, rectify_indices = torch.sort(
                topk_score, dim=-1, descending=True)

            # re-index
            topk_boxes = topk_boxes.gather(
                1, rectify_indices.unsqueeze(-1).expand_as(topk_boxes))
            topk_label = topk_label.gather(1, rectify_indices)
            topk_indices = topk_indices.gather(1, rectify_indices)

        # decode boxes into standard format
        topk_boxes = self.decode_box(topk_boxes, topk_indices)  # [B, K, 8]

        # box for nms (bird eye view)
        nms_boxes = topk_boxes[..., [0, 1, 3, 4, 6, 7]]

        # do nms for each sample
        pred_list = []
        batch_size = 1 if batch is None else batch['_info_']['size']
        for i in range(batch_size):
            keep_indices, valid_num = nms_bev(
                nms_boxes[i],
                topk_score[i],
                self.decode_cfg['nms_cfg']['overlap_thresh'],
                self.decode_cfg['nms_cfg']['post_num'],
            )
            keep_indices = keep_indices.long()
            det_box = topk_boxes[i][keep_indices]
            det_label = topk_label[i][keep_indices]
            det_score = topk_score[i][keep_indices]
            pred = (det_box, det_score, det_label, valid_num)
            pred_list.append(pred)

        return pred_list[0]

    def decode_trt(self, output, batch=None):
        pred_list = []
        batch_size = batch['_info_']['size']
        for i in range(batch_size):
            valid_num = output['valid'][i]
            boxes = output['boxes'][i][:valid_num].cpu().numpy()
            labels = output['label'][i][:valid_num].cpu().numpy()
            scores = output['score'][i][:valid_num].cpu().numpy()
            types = np.array([self.label_to_type[label]
                             for label in labels], dtype=np.int32)
            pred_list.append(Annotation3d(
                boxes=boxes, types=types, scores=scores))
        return pred_list

    def loss(self, output, batch):
        positive_index = batch['gt']['positive_indices'].long()
        positive_heatmap_index = batch['gt']['positive_heatmap_indices'].long()

        loss_dict = {}
        for head_name in self.loss_cfg['head_weight'].keys():
            head_prediction = output[head_name]
            if head_name == 'heatmap':
                heatmap_prediction = self.safe_sigmoid(head_prediction)
                loss = self.criteria_heatmap(
                    heatmap_prediction, batch['gt'][head_name], positive_heatmap_index)
            else:
                # prediction
                positive_prediction = head_prediction[positive_index[:, 0],
                                                      :,
                                                      positive_index[:, 1],
                                                      positive_index[:, 2]]

                # ground truth
                if head_name == 'iou':
                    # shape Nx8
                    gt_encoded_boxes = torch.cat([
                        batch['gt']['offset'],
                        batch['gt']['height'],
                        batch['gt']['size'],
                        batch['gt']['heading'],
                    ], dim=-1)
                    positive_gt_boxes = self.decode_box(gt_encoded_boxes.unsqueeze(
                        0), positive_index[:, 1:].unsqueeze(0)).squeeze(0)

                    pred_encoded_boxes = torch.cat([
                        output['offset'],
                        output['height'],
                        output['size'],
                        output['heading'],
                    ], dim=1).detach()  # !!! detach here
                    pred_encoded_boxes = pred_encoded_boxes[positive_index[:, 0],
                                                            :,
                                                            positive_index[:, 1],
                                                            positive_index[:, 2]]
                    positive_pred_boxes = self.decode_box(
                        pred_encoded_boxes.unsqueeze(0), positive_index[:, 1:].unsqueeze(0)).squeeze(0)

                    positive_pred_boxes_bev = positive_pred_boxes[:, [
                        0, 1, 3, 4, 6, 7]]
                    positive_gt_boxes_bev = positive_gt_boxes[:, [
                        0, 1, 3, 4, 6, 7]]
                    positive_gt_iou = iou_bev(
                        positive_pred_boxes_bev, positive_gt_boxes_bev)

                    # encode iou to range [-1, 1]
                    positive_gt = 2 * (positive_gt_iou.unsqueeze(-1) - 0.5)
                else:
                    positive_gt = batch['gt'][head_name]

                # special handling of heading
                if head_name == 'heading':
                    if self.free_heading_label is not None:
                        mask = positive_heatmap_index[:,
                                                      1] != self.free_heading_label
                        positive_prediction = positive_prediction[mask]
                        positive_gt = positive_gt[mask]
                    if self.normlize_rot:
                        mask = positive_prediction[:, 0] < 0
                        positive_prediction[mask] = -positive_prediction[mask]
                        positive_gt[mask] = -positive_gt[mask]

                loss = self.criteria_regression(
                    positive_prediction, positive_gt)
            loss_dict[f'loss_{head_name}'] = loss

        # calc the total loss of different heads
        assert(len(self.loss_cfg['head_weight']) == len(loss_dict))
        loss = 0
        for head_name in self.loss_cfg['head_weight'].keys():
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
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt

        fig = plt.figure()

        TypePalette = ['red', 'green', 'blue']

        type_num = sample['gt']['heatmap'].shape[0]
        for label in range(type_num):
            ax = fig.add_subplot(1, type_num, label+1, aspect='equal')

            if show_pcd:
                pcd = sample['input']['pcd']
                ax.scatter(pcd[:, 0], pcd[:, 1], c='black',
                           marker='.', alpha=0.7, s=1)

            if show_heatmap_gt:
                heatmap = sample['gt']['heatmap'][label]
                heatmap = torch.sigmoid(heatmap)
                x = self.point_range[0] + self.grid_size[0] * \
                    np.arange(self.grid_reso[0]+1)
                y = self.point_range[1] + self.grid_size[1] * \
                    np.arange(self.grid_reso[1]+1)
                ax.pcolormesh(x, y, heatmap, cmap='hot', alpha=0.6)

            if show_heatmap_pred:
                heatmap = sample['output']['heatmap'][label]
                heatmap = torch.sigmoid(heatmap)
                x = self.point_range[0] + self.grid_size[0] * \
                    np.arange(self.grid_reso[0]+1)
                y = self.point_range[1] + self.grid_size[1] * \
                    np.arange(self.grid_reso[1]+1)
                ax.pcolormesh(x, y, heatmap, cmap='hot', alpha=0.6)

            if show_box:
                positive_indices = sample['gt']['positive_indices']
                positive_labels = sample['gt']['positive_heatmap_indices'][:, 0]
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
                        (pos[i, 0].item(), pos[i, 1].item()), l[i].item(), w[i].item(), angle=rot[i], linewidth=1, edgecolor=TypePalette[positive_labels[i] % len(TypePalette)], facecolor='none')
                    ax.add_patch(rect)

        plt.show()

    def safe_sigmoid(self, x):
        return torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)

    def decode_box(self, boxes, cords):
        grid_offset = self.point_range[:2] + self.grid_size[:2]/2
        grid_offset = torch.from_numpy(grid_offset).to(boxes.device)
        grid_size = torch.from_numpy(self.grid_size[:2]).to(boxes.device)
        grid_reso = torch.from_numpy(self.grid_reso[:2]).to(boxes.device)

        # in case of empty boxes
        if boxes.shape[0] * boxes.shape[1] == 0:
            return boxes.new_empty((boxes.shape[0], boxes.shape[1], 8))

        # if is B x N, convert to B x N x 2
        if cords.dim() == 2:
            cords_y = torch.div(cords, grid_reso[0], rounding_mode='trunc')
            #  cords_x = cords % grid_reso[0]
            # workaround for tensorrt
            cords_x = cords - (cords_y * grid_reso[0])
            cords = torch.stack([cords_x, cords_y], dim=-1)

        grid_center = cords * grid_size + grid_offset

        decoded_box_xy = boxes[..., :2] + grid_center

        decoded_box_z = boxes[..., [2]]

        decoded_box_size = torch.exp(boxes[..., 3:6]) / 2

        decoded_box_heading = boxes[..., 6:] / \
            torch.norm(boxes[..., 6:], dim=-1, keepdim=True)

        decoded_box = torch.cat([
            decoded_box_xy,
            decoded_box_z,
            decoded_box_size,
            decoded_box_heading,
        ], dim=-1)

        return decoded_box

    def get_export_info(self, batch):
        input = (batch['input']['points'][0], )
        input_name = ['points']
        output_name = ['boxes', 'score', 'label', 'valid']
        dynamic_axes = {'points': {0: 'point_num'}}
        return input, input_name, output_name, dynamic_axes
