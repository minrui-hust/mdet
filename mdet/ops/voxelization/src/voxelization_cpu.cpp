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
                     const int num_points,
                     const int num_features,
                     torch::TensorAccessor<T_int, 3> voxel_coord_to_idx,
                     torch::TensorAccessor<T, 3> voxels,
                     torch::TensorAccessor<T_int, 2> coords,
                     torch::TensorAccessor<T_int, 1> point_num,
                     int* voxel_num
                     ) { // clang-format on
  /*
  // declare a temp coors
  at::Tensor temp_coors = at::zeros(
      {num_points, NDim}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));

  // First use dynamic voxelization to get coors,
  // then check max points/voxels constraints

  int voxelidx, num;
  auto coor = temp_coors.accessor<int, 2>();

  for (int i = 0; i < num_points; ++i) {
    // T_int* coor = temp_coors.data_ptr<int>() + i * NDim;

    if (coor[i][0] == -1)
      continue;

    voxelidx = coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]];

    // record voxel
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (max_voxels != -1 && voxel_num >= max_voxels)
        continue;
      voxel_num += 1;

      coor_to_voxelidx[coor[i][0]][coor[i][1]][coor[i][2]] = voxelidx;

      for (int k = 0; k < NDim; ++k) {
        coors[voxelidx][k] = coor[i][k];
      }
    }

    // put points into voxel
    num = num_points_per_voxel[voxelidx];
    if (max_points == -1 || num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels[voxelidx][num][k] = points[i][k];
      }
      num_points_per_voxel[voxelidx] += 1;
    }
  }

  return;
  */
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
  const int num_points = points.size(0);
  const int num_features = points.size(1);

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
            num_points,
            num_features,
            voxel_coord_to_idx.accessor<int, 3>(),
            voxels.accessor<scalar_t, 3>(),
            coords.accessor<int, 2>(), 
            point_num.accessor<int, 1>(),
            voxel_num.data_ptr<int>()); // clang-format on
      });
}

} // namespace voxelization
