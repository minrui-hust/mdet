#pragma once
#include <torch/extension.h>

typedef enum {
  NONE = 0,
  SUM,
  MEAN,
  MAX,
} reduce_t;

namespace voxelization {

// clang-format off
void voxelize_cpu(const at::Tensor &points,
                 const std::vector<float> &point_range,
                 const std::vector<float> &voxel_size,
                 const int max_points, 
                 const int max_voxels,
                 const reduce_t reduce_type,
                 at::Tensor &voxels, 
                 at::Tensor &coords,
                 at::Tensor &point_num,
                 at::Tensor &voxel_num);

void voxelize_gpu(const at::Tensor &points,
                 const std::vector<float> &point_range,
                 const std::vector<float> &voxel_size,
                 const int max_points, 
                 const int max_voxels,
                 const reduce_t reduce_type,
                 at::Tensor &voxels, 
                 at::Tensor &coords,
                 at::Tensor &point_num,
                 at::Tensor &voxel_num);
// clang-format on

inline reduce_t convert_reduce_type(const std::string &reduce_type) {
  if (reduce_type == "max")
    return reduce_t::MAX;
  else if (reduce_type == "sum")
    return reduce_t::SUM;
  else if (reduce_type == "mean")
    return reduce_t::MEAN;
  else if (reduce_type == "none")
    return reduce_t::NONE;
  else
    TORCH_CHECK(false, "do not support reduce type " + reduce_type)
  return reduce_t::NONE;
}

// Interface for Python
// clang-format off
inline void __Voxelize(const at::Tensor &points,
                      const std::vector<float> &point_range,
                      const std::vector<float> &voxel_size,
                      const int max_points, 
                      const int max_voxels,
                      const std::string& reduce_type,
                      at::Tensor &voxels, 
                      at::Tensor &coords,
                      at::Tensor &point_num,
                      at::Tensor &voxel_num) { // clang-format on
  if (points.device().is_cuda()) {
    return voxelize_gpu(points, point_range, voxel_size, max_points, max_voxels,
                        convert_reduce_type(reduce_type), voxels, coords,
                        point_num, voxel_num);
  } else {
    return voxelize_cpu(points, point_range, voxel_size, max_points, max_voxels,
                        convert_reduce_type(reduce_type), voxels, coords,
                        point_num, voxel_num);
  }
}

} // namespace voxelization
