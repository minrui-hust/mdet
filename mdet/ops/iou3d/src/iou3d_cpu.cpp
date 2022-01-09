#include "iou3d.h"

namespace iou3d {

void nms_bev_cpu(const at::Tensor &boxes, const at::Tensor &scores,
                 const float thresh, const int max_out, at::Tensor &selected,
                 at::Tensor &valid_num) {
  // TODO:
  assert(false);
}

} // namespace iou3d
