import torch
import torch.nn.functional as F


def focal_loss(predict, target, positive_index, alpha=2.0, beta=4.0):
    r'''
    focal loss same as "Object as Point"
    predict, target: B x C x H x W
    positive_index: N x 4, index of positive grid in heatmap
    alpha: see "Object as Point"
    beta: see "Object as Point"
    '''

    negtive_loss = -(torch.pow(predict, alpha) * torch.pow(1 -
                                                           target, beta) * torch.log(1 - predict)).sum()

    positive_prediction = predict[positive_index[:, 0],
                                  positive_index[:, 1], positive_index[:, 2], positive_index[:, 3]]

    positive_loss = -(torch.pow(1 - positive_prediction, alpha)
                      * torch.log(positive_prediction)).sum()

    if positive_index.size(0) == 0:
        return negtive_loss
    else:
        return (positive_loss + negtive_loss) / positive_index.size(0)


def regression_loss(predict, target):
    r'''
    predict, target: N x D
    '''
    loss = F.l1_loss(predict, target, reduction='none')
    return loss
