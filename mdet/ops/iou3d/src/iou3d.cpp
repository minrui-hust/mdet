#include "iou3d.h"

namespace iou3d {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("OpNMSBEV", &OpNMSBEV, "oriented nms in BEV");
}

} // namespace iou3d
