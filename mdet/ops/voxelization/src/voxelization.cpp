#include "voxelization.h"
#include <torch/extension.h>

namespace voxelization {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("__Voxelize", &__Voxelize, "voxelization");
}

} // namespace voxelization
