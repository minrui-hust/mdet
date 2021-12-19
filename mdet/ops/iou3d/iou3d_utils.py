import torch
from torch.autograd import Function

from . import iou3d


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    iou3d.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(),
                                 ans_iou)

    return ans_iou


class _nms_gpu(Function):
    @staticmethod
    def forward(ctx, boxes, scores, thresh, pre_maxsize=0, post_max_size=0):
        """Nms function with gpu implementation.

        Args:
            boxes (torch.Tensor): Input boxes with the shape of [N, 5]
                ([x1, y1, x2, y2, ry]).
            scores (torch.Tensor): Scores of boxes with the shape of [N].
            thresh (int): Threshold.
            pre_maxsize (int): Max size of boxes before nms. Default: None.
            post_maxsize (int): Max size of boxes after nms. Default: None.

        Returns:
            torch.Tensor: Indexes after nms.
        """
        order = scores.sort(0, descending=True)[1]
        if pre_maxsize > 0:
            order = order[:pre_maxsize]
        boxes = boxes[order].contiguous()

        keep = torch.zeros(boxes.size(0), dtype=torch.long)
        num_out = iou3d.nms_gpu(boxes, keep, thresh, boxes.device.index)
        keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
        if post_max_size > 0:
            keep = keep[:post_max_size]
        return keep

    @staticmethod
    def symbolic(g, boxes, scores, thresh, pre_maxsize=0, post_max_size=0):
        return g.op('custom_ops::nms_gpu', boxes, scores, thresh_f=thresh, pre_maxsize_i=pre_maxsize, post_max_size_i=post_max_size, outputs=1)


nms_gpu = _nms_gpu.apply


class _nms_normal_gpu(Function):
    @staticmethod
    def forward(ctx, boxes, scores, thresh):
        """Normal non maximum suppression on GPU.

        Args:
            boxes (torch.Tensor): Input boxes with shape (N, 5).
            scores (torch.Tensor): Scores of predicted boxes with shape (N).
            thresh (torch.Tensor): Threshold of non maximum suppression.

        Returns:
            torch.Tensor: Remaining indices with scores in descending order.
        """
        order = scores.sort(0, descending=True)[1]

        boxes = boxes[order].contiguous()

        keep = torch.zeros(boxes.size(0), dtype=torch.long)
        num_out = iou3d.nms_normal_gpu(boxes, keep, thresh,
                                            boxes.device.index)
        return order[keep[:num_out].cuda(boxes.device)].contiguous()

    @staticmethod
    def symbolic(g, boxes, scores, thresh):
        return g.op('custom_ops::nms_normal_gpu', boxes, scores, thresh_f=thresh, outputs=1)


nms_normal_gpu = _nms_normal_gpu.apply


class _nms(Function):
    @staticmethod
    def forward(ctx, boxes, scores, max_out, thresh):
        """Nms function with gpu implementation.

        Args:
            boxes (torch.Tensor): Input boxes with the shape of [N, 5]
                ([x1, y1, x2, y2, ry]).
            scores (torch.Tensor): Scores of boxes with the shape of [N].
            thresh (int): Threshold.
            pre_maxsize (int): Max size of boxes before nms. Default: None.
            post_maxsize (int): Max size of boxes after nms. Default: None.

        Returns:
            keep: torch.Tensor: Indexes after nms.
            num: first num of keep is valid
        """

        order = scores.sort(0, descending=True)[1]
        boxes = boxes[order].contiguous()

        keep = torch.zeros(boxes.size(0), dtype=torch.long)
        num_out = iou3d.nms_gpu(boxes, keep, thresh, boxes.device.index)
        if num_out > max_out:
            num_out = max_out
        keep = order[keep[:max_out].cuda(boxes.device)].contiguous()
        num = torch.tensor(num_out, dtype=torch.int32, device=boxes.device)

        return keep, num

    @staticmethod
    def symbolic(g, boxes, scores, max_out, thresh):
        return g.op('custom_ops::NMSBEV', boxes, scores, max_out_i=max_out, thresh_f=thresh, outputs=2)

nms = _nms.apply
