#pragma once
#include <iostream>
#include <torch/extension.h>

typedef enum {
  NONE = 0,
  SUM = 1,
  MEAN = 2,
  MAX = 3,
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

// Interface for Python
// clang-format off
inline void OpVoxelization(const at::Tensor &points,
                           const std::vector<float> &point_range,
                           const std::vector<float> &voxel_size,
                           const int max_points, 
                           const int max_voxels,
                           const int reduce_type,
                           at::Tensor &voxels, 
                           at::Tensor &coords,
                           at::Tensor &point_num,
                           at::Tensor &voxel_num) { // clang-format on
  if (points.device().is_cuda()) {
    return voxelize_gpu(points, point_range, voxel_size, max_points, max_voxels,
                        (reduce_t)reduce_type, voxels, coords, point_num,
                        voxel_num);
  } else {
    return voxelize_cpu(points, point_range, voxel_size, max_points, max_voxels,
                        (reduce_t)reduce_type, voxels, coords, point_num,
                        voxel_num);
  }
}

} // namespace voxelization
