#include "iou3d.h"

namespace iou3d {

void nms_bev_cpu(const at::Tensor &boxes, const at::Tensor &scores,
                 const float thresh, const int max_out, at::Tensor &selected,
                 at::Tensor &valid_num) {
  // TODO:
  assert(false);
}

void iou_bev_cpu(const at::Tensor &pboxes, const at::Tensor &qboxes,
                 at::Tensor &iou) {
  // TODO:
  assert(false);
}

} // namespace iou3d
