#include "voxelization.h"
#include <ATen/TensorUtils.h>
#include <torch/extension.h>

namespace {

constexpr size_t NDim = 3;

// clang-format off
template <typename T, typename T_int>
void voxelize_kernel(const torch::TensorAccessor<T, 2> points,
                     const std::vector<float>& point_range,
                     const std::vector<float>& voxel_size,
                     const std::vector<int>& voxel_reso,
                     const int max_points,
                     const int max_voxels,
                     const reduce_t reduce_type,
                     torch::TensorAccessor<T_int, 3> voxel_coord_to_idx,
                     torch::TensorAccessor<T, 3> voxels,
                     torch::TensorAccessor<T_int, 2> coords,
                     torch::TensorAccessor<T_int, 1> point_num,
                     int* voxel_num
                     ) { // clang-format on

  // pos -> cord ->idx
  std::array<int, NDim> voxel_coord;
  for (auto i = 0; i < points.size(0); ++i) {
    // calc the voxel coordiantes
    bool failed = false;
    for (auto j = 0u; j < NDim; ++j) {
      int c = floor((points[i][j] - point_range[j]) / voxel_size[j]);
      if (c < 0 || c >= voxel_reso[j]) {
        failed = true;
        break;
      }
      voxel_coord[NDim - 1 - j] = c;
    }
    if (failed) {
      continue;
    }

    // get voxel index
    auto &voxel_idx =
        voxel_coord_to_idx[voxel_coord[0]][voxel_coord[1]][voxel_coord[2]];
    if (voxel_idx == -1) {              // new voxel may need to create
      if ((*voxel_num) >= max_voxels) { // max voxel num reached
        continue;
      } else {
        voxel_idx = *voxel_num;
        ++(*voxel_num);
      }
    }

    auto &voxel_point_num = point_num[voxel_idx];
    if (voxel_point_num >= max_points) { // max point num of this voxel reached
      continue;
    } else { // copy point into voxel and increase the point_num of this voxel
      voxels[voxel_idx][voxel_point_num] = points[i];
      ++voxel_point_num;
    }
  }
}

} // namespace

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
                  at::Tensor &voxel_num){ // clang-format on

  // check device
  AT_ASSERTM(points.device().is_cpu(), "points must be a CPU tensor");

  std::vector<int> voxel_reso(NDim);
  for (auto i = 0u; i < NDim; ++i) {
    voxel_reso[i] =
        round((point_range[NDim + i] - point_range[i]) / voxel_size[i]);
  }

  // indexed by z,y,x
  at::Tensor voxel_coord_to_idx = -at::ones(
      {voxel_reso[2], voxel_reso[1], voxel_reso[0]}, coords.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "voxelize_cpu", [&] {
        voxelize_kernel<scalar_t, int>( // clang-format off
            points.accessor<scalar_t, 2>(), 
            point_range,
            voxel_size,
            voxel_reso,
            max_points,
            max_voxels,
            reduce_type,
            voxel_coord_to_idx.accessor<int, 3>(),
            voxels.accessor<scalar_t, 3>(),
            coords.accessor<int, 2>(), 
            point_num.accessor<int, 1>(),
            voxel_num.data_ptr<int>()); // clang-format on
      });
}

} // namespace voxelization
