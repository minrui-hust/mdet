#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include "voxelization.h"
#include "voxelization_kernel.h"

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_DATATYPE(x, dtype)                                               \
  TORCH_CHECK(x.scalar_type() == dtype, #x " must be float")

#define CHECK_INPUT(x, dtype)                                                  \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_DATATYPE(x, dtype)

namespace {
template <typename T_int> inline T_int CeilDiv(T_int a, T_int b) {
  return (a + b - 1) / b;
}
} // namespace

namespace voxelization {

void voxelize_2d( // clang-format off
  const at::Tensor &points,
  const std::vector<float> &point_range,
  const std::vector<float> &voxel_size,
  const std::vector<int32_t> &voxel_reso,
  const int max_points, 
  const int max_voxels,
  const reduce_t reduce_type,
  at::Tensor &voxels, 
  at::Tensor &coords,
  at::Tensor &voxel_points,
  at::Tensor &voxel_num
){ // clang-format on

  int point_num = points.size(0);
  int point_dim = points.size(1);

  const auto &max_pillars = max_voxels;

  // prepare initialized buffer
  at::Tensor map_key_to_list =
      coords.new_full({voxel_reso[0] * voxel_reso[1]}, -1);

  // prepare uninitialized buffer
  int uninit_bytes = 0;
  int point_nodes_bytes = point_num * sizeof(ListNode);
  uninit_bytes += point_nodes_bytes;

  int map_pillar_to_key_bytes = max_pillars * sizeof(int);
  uninit_bytes += map_pillar_to_key_bytes;

  int voxel_dists_bytes =
      reduce_type == reduce_t::NEAREST ? max_voxels * sizeof(float) : 0;
  uninit_bytes += voxel_dists_bytes;

  at::Tensor uninit_tensor = points.new_empty(
      {uninit_bytes}, points.options().dtype(at::ScalarType::Char));

  int8_t *point_nodes_data = uninit_tensor.data_ptr<int8_t>();
  int8_t *map_pillar_to_key_data = point_nodes_data + point_nodes_bytes;
  int8_t *voxel_dists_data = map_pillar_to_key_data + map_pillar_to_key_bytes;

  int block = 512;
  int grid;

  // clang-format off
  // point to pillar
  grid= CeilDiv(point_num, block);
  ScatterPointToPillar<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
    points.data_ptr<float>(), point_num,  point_dim,
    point_range[0], point_range[1], point_range[2],
    voxel_size[0],  voxel_size[1],  voxel_size[2],
    voxel_reso[0],  voxel_reso[1],  voxel_reso[2],
    max_pillars,
    (reduce_type==reduce_t::NEAREST), // whether record distance

    // output
    (ListNode*)point_nodes_data, // companion node of each point, same number as points
    (int*)map_key_to_list.data_ptr(), // point key to pillar id hash map, initialized to -1, size: pillar_reso_x*pillar_reso_y
    (int*)map_pillar_to_key_data, // pillar id to head node id of the list of this pillar, size: max_pillars, initialized to -1(if append_to_pillar set to true)
    (int*)voxel_num.data_ptr() // valid pillar number, init to 0
  );

  // gather voxel/pillar from points
  grid= CeilDiv(max_voxels, block);
  GatherVoxelFromPoint<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
    points.data_ptr<float>(), point_num, point_dim,
    (ListNode *) point_nodes_data,
    (int*)map_key_to_list.data_ptr(), // voxel id to head node id of the list of this voxel, size: max_voxels, filled by ScatterPointToVoxel
    (int*)map_pillar_to_key_data, // voxel id to head node id of the list of this voxel, size: max_voxels, filled by ScatterPointToVoxel
    (int*)voxel_num.data_ptr(),
    max_voxels,
    max_points,
    reduce_type,
    (float*)voxel_dists_data,
    voxels.data_ptr<float>(),
    coords.data_ptr<int>(),
    voxel_points.data_ptr<int>()
  );
  // clang-format on
}

void voxelize_3d( // clang-format off
  const at::Tensor &points,
  const std::vector<float> &point_range,
  const std::vector<float> &voxel_size,
  const std::vector<int32_t> &voxel_reso,
  const int max_points, 
  const int max_voxels,
  const reduce_t reduce_type,
  at::Tensor &voxels, 
  at::Tensor &coords,
  at::Tensor &voxel_points,
  at::Tensor &voxel_num
){ // clang-format on
  int point_num = points.size(0);
  int point_dim = points.size(1);

  int block = 512;
  int grid;

  // allocate memory
  at::Tensor map_key_to_pillar =
      coords.new_full({voxel_reso[0] * voxel_reso[1]}, -1);
  at::Tensor pillar_num = coords.new_zeros({});

  // uninit
  int uninit_bytes = 0;
  int point_nodes_bytes = point_num * sizeof(ListNode);
  uninit_bytes += point_nodes_bytes;

  int map_voxel_to_key_bytes = max_voxels * sizeof(int);
  uninit_bytes += map_voxel_to_key_bytes;

  int voxel_dists_bytes =
      reduce_type == reduce_t::NEAREST ? max_voxels * sizeof(float) : 0;
  uninit_bytes += voxel_dists_bytes;

  at::Tensor uninit_tensor = points.new_empty(
      {uninit_bytes}, points.options().dtype(at::ScalarType::Char));

  int8_t *point_nodes_data = uninit_tensor.data_ptr<int8_t>();
  int8_t *map_voxel_to_key_data = point_nodes_data + point_nodes_bytes;
  int8_t *voxel_dists_data = map_voxel_to_key_data + map_voxel_to_key_bytes;

  // clang-format off
  grid= CeilDiv(point_num, block);
  ConstructPointToPillarMap<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
    points.data_ptr<float>(), point_num,  point_dim,  // points,
    point_range[0], point_range[1], point_range[2],
    voxel_size[0],  voxel_size[1],  voxel_size[2],
    voxel_reso[0],  voxel_reso[1],  voxel_reso[2],
    (reduce_type==reduce_t::NEAREST), // whether record distance

    (ListNode *)point_nodes_data, // companion node of each point, same number as points
    (int *)map_key_to_pillar.data_ptr(), // key to node list
    (int *)pillar_num.data_ptr()// valid pillar number, init to 0
  );

  at::Tensor map_key_to_list = coords.new_full({pillar_num.item().toInt() * voxel_reso[2]}, -1);

  grid= CeilDiv(point_num, block);
  ScatterPointToVoxel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
    points.data_ptr<float>(), point_num, point_dim,  // points,
    (ListNode *) point_nodes_data, // node for each point, store tmp info of the point
    (int *)map_key_to_pillar.data_ptr(),
    voxel_reso[2],
    max_voxels,
  
    // output
    (int *)map_key_to_list.data_ptr(), // point key to voxel id, init to -1, size: pillar_num*voxel_reso_z
    (int *)map_voxel_to_key_data, // voxel to voxel's first point, init to -1, size: max_voxels
    (int *)voxel_num.data_ptr()// output valid voxel number, init to 0
  );

  // gather voxel/pillar from points
  grid= CeilDiv(max_voxels, block);
  GatherVoxelFromPoint<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
    points.data_ptr<float>(), point_num, point_dim,
    (ListNode *) point_nodes_data,
    (int*)map_key_to_list.data_ptr(), // voxel id to head node id of the list of this voxel, size: max_voxels, filled by ScatterPointToVoxel
    (int*)map_voxel_to_key_data, // voxel id to head node id of the list of this voxel, size: max_voxels, filled by ScatterPointToVoxel
    (int*)voxel_num.data_ptr(),
    max_voxels,
    max_points,
    reduce_type,
    (float*)voxel_dists_data,
    voxels.data_ptr<float>(),
    coords.data_ptr<int>(),
    voxel_points.data_ptr<int>()
  );
  // clang-format on
}

// clang-format off
void voxelize_gpu(
  const at::Tensor &points,
  const std::vector<float> &point_range,
  const std::vector<float> &voxel_size,
  const std::vector<int32_t> &voxel_reso,
  const int max_points, 
  const int max_voxels,
  const reduce_t reduce_type,
  at::Tensor &voxels, 
  at::Tensor &coords,
  at::Tensor &voxel_points,
  at::Tensor &voxel_num
){
  // clang-format on
  CHECK_INPUT(points, at::ScalarType::Float);
  CHECK_INPUT(voxels, at::ScalarType::Float);
  CHECK_INPUT(coords, at::ScalarType::Int);
  CHECK_INPUT(voxel_points, at::ScalarType::Int);
  CHECK_INPUT(voxel_num, at::ScalarType::Int);

  at::cuda::CUDAGuard device_guard(points.device());

  // do some init
  voxel_num.fill_(0);

  if (voxel_reso.size() < 3 || voxel_reso[2] <= 1) {
    voxelize_2d(points, point_range, voxel_size, voxel_reso, max_points,
                max_voxels, reduce_type, voxels, coords, voxel_points,
                voxel_num);
  } else {
    voxelize_3d(points, point_range, voxel_size, voxel_reso, max_points,
                max_voxels, reduce_type, voxels, coords, voxel_points,
                voxel_num);
  }

  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace voxelization
