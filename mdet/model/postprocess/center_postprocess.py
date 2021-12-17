import torch
import torch.nn as nn
import torch.nn.functional as F
from mdet.utils.factory import FI
from mdet.model import BaseModule
import mdet.model.loss.loss as loss
from functools import partial


@FI.register
class CenterPostProcess(BaseModule):
    def __init__(self, head_weight, alpha, beta):
        super().__init__()

        self.head_weight = head_weight

        self.criteria_heatmap = partial(
            loss.focal_loss, alpha=alpha, beta=beta)
        self.criteria_regression = partial(F.l1_loss, reduction='mean')

    def forward_train(self, result, batch):
        r'''
        forward_train calc loss
        '''

        positive_index = batch['gt']['positive_indices'].long()
        positive_class = batch['gt']['categories'].long()
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
        assert(len(self.head_weight) == len(loss_dict))
        loss = 0
        for head_name in self.head_weight:
            head_loss = loss_dict[f'loss_{head_name}']
            head_wgt = self.head_weight[head_name]
            loss += head_wgt*head_loss

        loss_dict['loss'] = loss

        return loss_dict

    def forward_eval(self, result, batch):
        r'''
        forward_eval decode for evaluation
        '''
        return result

    def forward_infer(self, result, batch):
        r'''
        forward_eval decode for deployment
        '''
        raise NotImplementedError

    def safe_sigmoid(self, x):
        return torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)
