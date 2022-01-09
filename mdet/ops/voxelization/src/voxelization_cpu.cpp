#include "voxelization.h"
#include <ATen/TensorUtils.h>
#include <torch/extension.h>

namespace voxelization {

constexpr size_t NDim = 3;

// clang-format off
template <typename T, typename T_int>
void voxelize_kernel(const torch::TensorAccessor<T, 2> points,
                     const std::vector<float>& point_range,
                     const std::vector<float>& voxel_size,
                     const std::vector<int32_t>& voxel_reso,
                     const int max_points,
                     const int max_voxels,
                     const reduce_t reduce_type,
                     torch::TensorAccessor<T_int, 3> voxel_coord_to_idx,
                     torch::TensorAccessor<T, 3> voxels,
                     torch::TensorAccessor<T_int, 2> coords,
                     torch::TensorAccessor<T_int, 1> point_num,
                     int* voxel_num
                     ) { // clang-format on
  // point feature size
  auto FDim = points.size(-1);

  // used for reduce_t::NEAREST
  std::vector<float> voxel_dists;
  if (reduce_type == reduce_t::NEAREST) {
    voxel_dists.resize(max_voxels, std::numeric_limits<float>::max());
  }

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
    if (voxel_idx == -1) {
      if ((*voxel_num) >= max_voxels) { // max voxel num reached
        continue;
      } else { // new voxel
        voxel_idx = *voxel_num;
        ++(*voxel_num);
        for (auto j = 0u; j < NDim; ++j) {
          coords[voxel_idx][j] = voxel_coord[j];
        }
      }
    }

    // add point into voxel
    auto &voxel_point_num = point_num[voxel_idx];
    if (voxel_point_num >= max_points &&
        max_points != 0) { // max point num of this voxel reached
      continue;
    } else { // copy point into voxel and increase the point_num of this voxel
      if (reduce_type == reduce_t::MEAN) {
        for (auto j = 0u; j < FDim; ++j) {
          voxels[voxel_idx][0][j] +=
              (points[i][j] - voxels[voxel_idx][0][j]) / (voxel_point_num + 1);
        }
      } else if (reduce_type == reduce_t::NEAREST) {
        // calc point to origin distance
        float point_dist = 0;
        for (auto j = 0u; j < NDim; ++j) {
          point_dist += points[i][j] * points[i][j];
        }

        // get voxel to origin distance
        auto &voxel_dist = voxel_dists[voxel_idx];

        // update voxel to origin distance accordingly
        if (point_dist < voxel_dist) {
          for (auto j = 0u; j < FDim; ++j) {
            voxels[voxel_idx][0][j] = points[i][j];
          }
          voxel_dist = point_dist;
        }
      } else { // NONE, FIRST
        for (auto j = 0u; j < FDim; ++j) {
          voxels[voxel_idx][voxel_point_num][j] = points[i][j];
        }
      }
    }
    ++voxel_point_num;
  }
}

// clang-format off
void voxelize_cpu(const at::Tensor &points,
                  const std::vector<float> &point_range,
                  const std::vector<float> &voxel_size,
                  const std::vector<int32_t> &voxel_reso,
                  const int max_points, 
                  const int max_voxels,
                  const reduce_t reduce_type,
                  at::Tensor &voxels, 
                  at::Tensor &coords,
                  at::Tensor &point_num,
                  at::Tensor &voxel_num){ // clang-format on

  // check device
  AT_ASSERTM(points.device().is_cpu(), "points must be a CPU tensor");

  // indexed by z,y,x, initialize to -1
  at::Tensor voxel_coord_to_idx = at::full(
      {voxel_reso[2], voxel_reso[1], voxel_reso[0]}, -1, coords.options());

  point_num.fill_(0);
  voxel_num.fill_(0);

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
