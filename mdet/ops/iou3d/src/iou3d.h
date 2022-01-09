#pragma once
#include <torch/extension.h>

namespace iou3d {

void nms_bev_cpu(const at::Tensor &boxes, const at::Tensor &scores,
                 const float thresh, const int max_out, at::Tensor &selected,
                 at::Tensor &valid_num);

void nms_bev_gpu(const at::Tensor &boxes, const at::Tensor &scores,
                 const float thresh, const int max_out, at::Tensor &selected,
                 at::Tensor &valid_num);

inline void OpNMSBEV(const at::Tensor &boxes, const at::Tensor &scores,
                     const float thresh, const int max_out,
                     at::Tensor &selected, at::Tensor &valid_num) {
  if (boxes.device().is_cuda()) {
    return nms_bev_gpu(boxes, scores, thresh, max_out, selected, valid_num);
  } else {
    return nms_bev_cpu(boxes, scores, thresh, max_out, selected, valid_num);
  }
}

} // namespace iou3d
