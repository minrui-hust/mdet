import torch
import torch.nn as nn
import torch.nn.functional as F
from mdet.utils.factory import FI
import mdet.model.loss.loss as loss
from functools import partial
from mdet.core.annotation import Annotation3d
from mdet.ops.iou3d import nms
from mdet.model.postprocess.base_postprocess import BasePostProcess


@FI.register
class CenterPostProcess(BasePostProcess):
    def __init__(self, loss_cfg={}, nms_cfg={}, codec=None):
        super().__init__()

        self.loss_cfg = loss_cfg

        self.nms_cfg = nms_cfg

        self.codec = FI.create(codec)

        self.criteria_heatmap = partial(
            loss.focal_loss, alpha=loss_cfg['alpha'], beta=loss_cfg['beta'])
        self.criteria_regression = partial(F.l1_loss, reduction='mean')

    def loss(self, result, batch):
        r'''
        forward_train calc loss
        '''

        positive_index = batch['gt']['positive_indices'].long()
        positive_class = batch['gt']['types'].long()
        positive_heatmap_index = torch.cat(
            [positive_index[:, [0]], positive_class.unsqueeze(-1), positive_index[:, 1:]], dim=-1)

        loss_dict = {}
        for head_name, head_prediction in result.items():
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

    @torch.no_grad()
    def eval(self, result):
        r'''
        decode, nms, return detection in standard format, which Annotation3d
        input:
            result:
            batch:
        output: 
            list of Annotation3d
        '''

        heatmap = torch.sigmoid(result['heatmap'])
        B, C, H, W = heatmap.shape

        score, label = heatmap.max(dim=1)
        score = score.reshape(-1, H*W)  # [B, H*W]
        label = label.reshape(-1, H*W)  # [B, H*W]

        boxes = torch.cat([
            result['offset'],
            result['height'],
            result['size'],
            result['heading'],
        ], dim=1)
        boxes = boxes.permute(0, 2, 3, 1).reshape(-1, H*W, 8)  # [B, H*W, 8]

        # topk_socre: [B, K], topk_indices: [B, K], K = pre_nms_num
        topk_score, topk_indices = score.topk(self.nms_cfg['pre_num'], dim=1)

        topk_boxes = boxes.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, boxes.size(2)))  # [B, K, 8]
        topk_label = label.gather(1, topk_indices)  # [B, K]

        # decode boxes into standard format
        topk_boxes = self.codec.decode(topk_boxes, topk_indices)  # [B, 7]

        # do nms for each sample
        output = []
        for i in range(B):
            # keep_indices: [k], valid_num: [0], k=post_nms_num
            keep_indices, valid_num = nms(
                topk_boxes[i], topk_score[i], self.nms_cfg['post_num'], self.nms_cfg['overlap_thresh'])
            valid_indices = keep_indices[:valid_num]
            det_box = topk_boxes[i][valid_indices]
            det_label = topk_label[i][valid_indices]
            det_score = topk_score[i][valid_indices]
            output.append(Annotation3d(
                boxes=det_box, types=det_label, scores=det_score))

        return output

    @torch.no_grad()
    def infer(self, result):
        r'''
        forward_eval decode for deployment
        '''
        self.eval(result)

    def safe_sigmoid(self, x):
        return torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)
