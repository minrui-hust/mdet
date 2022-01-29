import torch
from torch.autograd import Function
from torch.cuda.amp.autocast_mode import custom_fwd

from .iou3d import OpNMSBEV


class _nms_bev(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, boxes, scores, thresh, max_out):
        """Nms function with gpu implementation.

        Args:
            boxes (torch.Tensor): Input boxes with the shape of [N, 6]
                ([center.x, center.y, extend.x, extend.y, cos(alpha), sin(alpha)]).
            scores (torch.Tensor): Scores of boxes with the shape of [N]. in descending order
            thresh (int): Threshold.
            max_out (int): Max size of boxes after nms. Default: None.

        Returns:
            selected: torch.Tensor: Indexes after nms.
            valid_num: first valid_num of selected is valid
        """

        selected = boxes.new_zeros(max_out, dtype=torch.int32)
        valid_num = boxes.new_tensor(0, dtype=torch.int32)

        OpNMSBEV(boxes, scores, thresh, max_out, selected, valid_num)

        return selected, valid_num

    @staticmethod
    def symbolic(g, boxes, scores, thresh, max_out):
        return g.op('custom_ops::NMSBEV',
                    boxes,
                    scores,
                    thresh_f=thresh,
                    max_out_i=max_out,
                    outputs=2)


nms_bev = _nms_bev.apply
