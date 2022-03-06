import torch
import torch.nn.functional as F


def focal_loss(predict, target, positive_index, alpha=2.0, beta=4.0, full_positive_loss=False):
    r'''
    focal loss same as "Object as Point"
    predict, target: B x C x H x W
    positive_index: N x 4, index of positive grid in heatmap
    alpha: see "Object as Point"
    beta: see "Object as Point"
    '''

    negtive_loss = -(torch.pow(predict, alpha) * torch.pow(1 -
                                                           target, beta) * torch.log(1 - predict)).sum()

    if full_positive_loss:
        positive_loss = -(torch.pow(target, beta) *
                          torch.pow(1-predict, alpha)*torch.log(predict)).sum()
    else:
        positive_prediction = predict[positive_index[:, 0],
                                      positive_index[:, 1], positive_index[:, 2], positive_index[:, 3]]

        positive_loss = -(torch.pow(1 - positive_prediction, alpha)
                          * torch.log(positive_prediction)).sum()

    if positive_index.size(0) == 0:
        return negtive_loss
    else:
        return (positive_loss + negtive_loss) / positive_index.size(0)


def regression_loss(predict, target, eps=1e-4, reduce=True):
    r'''
    predict, target: N x D
    '''
    loss = F.l1_loss(predict, target, reduction='sum')
    if reduce:
        loss = loss / (target.nelement() + eps)
    return loss
