#include "voxelization.h"
#include <torch/extension.h>

namespace voxelization {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("OpVoxelization", &OpVoxelization, "voxelization");
}

} // namespace voxelization
