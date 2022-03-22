from functools import partial
import math
import os

from mai.data.codecs import BaseCodec
from mai.utils import FI
import numba as nb
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

from mdet.core.annotation3d import Annotation3d
from mai.model.loss import loss
from mdet.ops.iou3d import iou_bev, nms_bev


@FI.register
class DetrCodec(BaseCodec):
    r'''
    Assign cls, pos, height, size, cos_theta, sin_theta, positive_indices
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

        # many type may map to same label
        self.type_to_label = {}
        self.label_to_type = {}
        for label, type_list in encode_cfg['labels'].items():
            for type in type_list:
                self.type_to_label[type] = label
                # this may loss information, only record last type
                self.label_to_type[label] = type

        # for decode

        # for loss
        self.criteria_cls = nn.CrossEntropyLoss(reduction='mean')
        self.criteria_regression = partial(loss.regression_loss, eps=1e-4)

    def encode_data(self, sample, info):
        # just pcd for now
        sample['input'] = dict(points=torch.from_numpy(
            sample['data']['pcd'].points))

    def encode_anno(self, sample, info):
        boxes = sample['anno'].boxes
        types = sample['anno'].types
        labels = np.array([self.type_to_label[type]
                          for type in types], dtype=np.int32)

        # pos, normlize by point range
        pos = boxes[:, :2]  # / (self.point_range[3:5] - self.point_range[0:2])

        # height
        height = boxes[:, [2]]

        # size, in log format
        size = np.log(boxes[:, 3:6]*2)

        # heading, in complex number format
        heading = boxes[:, 6:]

        # keep gt in numpy for real gt is calculated based on prediction
        sample['gt'] = dict(pos=pos,
                            height=height,
                            size=size,
                            heading=heading,
                            labels=labels,
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
        gt_encoded_boxes = []
        gt_label = []
        for i in range(batch['_info_']['size']):
            gt_encoded_boxes.append(
                np.concatenate([
                    batch['gt'][i]['pos'],
                    batch['gt'][i]['height'],
                    batch['gt'][i]['size'],
                    batch['gt'][i]['heading'],
                ], axis=-1)
            )
            gt_label.append(batch['gt'][i]['labels'])

        pred_encoded_boxes = torch.cat([
            output['pos'],
            output['height'],
            output['size'],
            output['heading'],
        ], dim=-1).detach().cpu().numpy()
        pred_cls = torch.softmax(output['cls'], dim=-1).detach().cpu().numpy()

        B = batch['_info_']['size']
        N = pred_cls.shape[1]
        D = pred_encoded_boxes.shape[-1]
        device = output['pos'].device

        gt_padded_boxes = []
        gt_padded_label = []
        for i in range(batch['_info_']['size']):
            cost_matrix = box_cost_matrix(
                pred_encoded_boxes[i], gt_encoded_boxes[i], pred_cls[i], gt_label[i])
            matched_pred, matched_gt = linear_sum_assignment(cost_matrix)
            boxes = np.zeros((N, D), dtype=np.float32)
            boxes[matched_pred] = gt_encoded_boxes[i][matched_gt]
            gt_padded_boxes.append(boxes)

            label = np.zeros(N, dtype=np.int32)
            label[matched_pred] = gt_label[i][matched_gt]
            gt_padded_label.append(label)

        # make gt on the fly
        gt_padded_boxes = torch.from_numpy(
            np.concatenate(gt_padded_boxes, axis=0)).to(device)
        gt_padded_label = torch.from_numpy(
            np.concatenate(gt_padded_label, axis=0)).to(device)
        gt = dict(
            pos=gt_padded_boxes[:, :2],
            height=gt_padded_boxes[:, [2]],
            size=gt_padded_boxes[:, 3:6],
            heading=gt_padded_boxes[:, 6:],
            label=gt_padded_label,
        )

        # now we have real gt, calc the loss
        loss_dict = {}
        for head_name in self.loss_cfg['head_weight'].keys():
            head_prediction = output[head_name].view(B*N, -1)
            if head_name == 'cls':
                loss = self.criteria_cls(head_prediction, gt['label'].long())
            else:
                positive_mask = gt['label'] != 0
                positive_gt = gt[head_name][positive_mask]
                positive_prediction = head_prediction[positive_mask]
                if head_name == 'pos':
                    positive_prediction = positive_prediction*75.0
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
                '.gt': dict(type='append'),

                # rules for output, would be very simillar
                '.output.offset': dict(type='stack'),
                '.output.height': dict(type='stack'),
                '.output.size': dict(type='stack'),
                '.output.heading': dict(type='stack'),
                '.output.cls': dict(type='stack'),

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


@nb.njit(fastmath=True)
def box_cost_matrix(pred_boxes, gt_boxes, pred_cls, gt_label):
    pred_box_num = len(pred_boxes)
    gt_box_num = len(gt_boxes)

    assert(pred_box_num == len(pred_cls))
    assert(gt_box_num == len(gt_label))

    cost = np.empty((pred_box_num, gt_box_num), dtype=np.float32)

    for i in range(pred_box_num):
        for j in range(gt_box_num):
            cost[i, j] = np.mean(
                np.abs(pred_boxes[i, :2]*75.0 - gt_boxes[j, :2])) - pred_cls[i][gt_label[j]]
            #  cost[i, j] = np.linalg.norm(pred_boxes[i, :2] - gt_boxes[j, :2])

    return cost
