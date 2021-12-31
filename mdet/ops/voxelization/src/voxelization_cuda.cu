#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include "voxelization.h"
#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_DATATYPE(x)                                                      \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float")

#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x);                                                         \
  CHECK_DATATYPE(x)

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                   \
       i += blockDim.x * gridDim.x)

namespace {

template <typename T_int> inline T_int CeilDiv(T_int a, T_int b) {
  return (a + b - 1) / b;
}

// list all the point in a voxel
struct __align__(4) ListNode {
  int next;   // next point id
  int3 cord;  // voxel coord of this point
  float dist; // min voxel distance to origin
  int key;
};

__global__ void ScatterPointToPillar( // clang-format off
  // input
  const float* __restrict__ points, int point_num,  int point_dim,  // points,
  float range_min_x, float range_min_y, float range_min_z, // range
  float voxel_size_x, float voxel_size_y, float voxel_size_z, // voxel size
  int voxel_reso_x, int voxel_reso_y, int voxel_reso_z, // voxel resolution
  int max_pillars, // max pillar number
  bool record_dist, // whether record squared dist

  // output
  ListNode *point_nodes, // companion node of each point, same number as points
  int *map_key_to_list, // key to node list
  int *map_pillar_to_key, // pillar id to key of map_point_to_list
  int *valid_pillar_num // valid pillar number, init to 0
){ // clang-format on
  CUDA_1D_KERNEL_LOOP(i, point_num) {
    const auto &point_id = i;
    const float *cur_point = points + point_id * point_dim;
    ListNode *cur_node = point_nodes + point_id;

    // calc which pillar this point scatter to
    auto &cord = cur_node->cord;
    cord.x = floor((cur_point[0] - range_min_x) / voxel_size_x);
    cord.y = floor((cur_point[1] - range_min_y) / voxel_size_y);
    cord.z = floor((cur_point[2] - range_min_z) / voxel_size_z);

    // clang-format off
    if (cord.x < 0 || cord.x >= voxel_reso_x ||
        cord.y < 0 || cord.y >= voxel_reso_y ||
        cord.z < 0 || cord.z >= voxel_reso_z ){
      continue;
    }
    // clang-format on

    // store dist if needed
    if (record_dist) {
      cur_node->dist = cur_point[0] * cur_point[0] +
                       cur_point[1] * cur_point[1] +
                       cur_point[2] * cur_point[2];
    }

    int key = voxel_reso_x * cord.y + cord.x;

    cur_node->next = atomicExch(map_key_to_list + key, point_id);

    // should make a new node
    if (cur_node->next == -1) {
      int pillar_id = atomicAdd(valid_pillar_num, 1);
      if (pillar_id < max_pillars) {
        map_pillar_to_key[pillar_id] = key;
      }
    }
  }
}

__global__ void ConstructPointToPillarMap( // clang-format off
  // input
  const float* __restrict__ points, int point_num,  int point_dim,  // points,
  float range_min_x, float range_min_y, float range_min_z, // range
  float voxel_size_x, float voxel_size_y, float voxel_size_z, // voxel size
  int voxel_reso_x, int voxel_reso_y, int voxel_reso_z, // voxel resolution
  bool record_dist, // whether record squared dist

  // output
  ListNode *point_nodes, // companion node of each point, same number as points
  int *map_key_to_pillar, // key to node list
  int *pillar_num // valid pillar number, init to 0
){ // clang-format on
  CUDA_1D_KERNEL_LOOP(i, point_num) {
    const auto &point_id = i;
    const float *cur_point = points + point_id * point_dim;
    ListNode *cur_node = point_nodes + point_id;

    // calc which pillar this point scatter to
    auto &cord = cur_node->cord;
    cord.x = floor((cur_point[0] - range_min_x) / voxel_size_x);
    cord.y = floor((cur_point[1] - range_min_y) / voxel_size_y);
    cord.z = floor((cur_point[2] - range_min_z) / voxel_size_z);

    // clang-format off
    if (cord.x < 0 || cord.x >= voxel_reso_x ||
        cord.y < 0 || cord.y >= voxel_reso_y ||
        cord.z < 0 || cord.z >= voxel_reso_z ){
      cur_node->key = -1; // set invalide flag for out range point
      continue;
    } // clang-format on

    // store dist if needed
    if (record_dist) {
      cur_node->dist = cur_point[0] * cur_point[0] +
                       cur_point[1] * cur_point[1] +
                       cur_point[2] * cur_point[2];
    }

    cur_node->key = voxel_reso_x * cord.y + cord.x;

    if (-1 == atomicCAS(map_key_to_pillar + cur_node->key, -1, point_id)) {
      map_key_to_pillar[cur_node->key] = atomicAdd(pillar_num, 1);
    }
  }
}

__global__ void ScatterPointToVoxel( // clang-format off
  // input
  const float* __restrict__ points, int point_num, int point_dim,  // points,
  ListNode * __restrict__ point_nodes, // node for each point, store tmp info of the point
  const int * __restrict__ map_key_to_pillar,
  int voxel_reso_z,
  int max_voxels,

  // output
  int *map_key_to_list, // point key to voxel id, init to -1, size: pillar_num*voxel_reso_z
  int *map_voxel_to_key, // voxel to voxel's first point, init to -1, size: max_voxels
  int *valid_voxels// output valid voxel number, init to 0
){ // clang-format on
  CUDA_1D_KERNEL_LOOP(i, point_num) {
    const auto &point_id = i;
    ListNode *cur_node = point_nodes + point_id;
    const auto &pillar_key = cur_node->key;
    if (pillar_key == -1) { // this point is out of range
      continue;
    }

    int pillar_id = map_key_to_pillar[pillar_key];

    int key = voxel_reso_z * pillar_id + cur_node->cord.z;

    cur_node->next = atomicExch(map_key_to_list + key, point_id);

    if (cur_node->next == -1) {
      int voxel_id = atomicAdd(valid_voxels, 1);
      if (voxel_id < max_voxels) {
        map_voxel_to_key[voxel_id] = key;
      }
    }
  }
}

__global__ void GatherVoxelFromPoint( // clang-format off
  // input
  const float* __restrict__ points, int point_num, int point_dim, // points
  const ListNode * __restrict__ point_nodes, // point nodes store each point's info, filled by ScatterPointToVoxel
  const int * __restrict__ map_key_to_list,
  const int * __restrict__ map_voxel_to_key, // voxel id to head node id of the list of this voxel, size: max_voxels, filled by ScatterPointToVoxel
  int * __restrict__ valid_voxel_num, // valid voxel num
  int max_voxels, // max output voxels
  int max_points, // max number of points in a voxel
  reduce_t reduce_type, // how to reduce points in voxel

  // tmp
  float* voxel_dists,

  // output
  float *voxels,
  int* voxel_cords,
  int* voxel_points
) { // clang-format on
  CUDA_1D_KERNEL_LOOP(i, min(*valid_voxel_num, max_voxels)) {
    const auto &voxel_id = i;
    int voxel_point_num = reduce_type == reduce_t::NONE ? max_points : 1;
    float *cur_voxel = voxels + voxel_id * voxel_point_num * point_dim;
    int *cur_voxel_cord = voxel_cords + voxel_id * 3;
    float &cur_voxel_dist = voxel_dists[voxel_id];

    // the list head
    int cur_point_global_id = map_key_to_list[map_voxel_to_key[voxel_id]];

    // NOTE: in reverse order
    cur_voxel_cord[0] = point_nodes[cur_point_global_id].cord.z;
    cur_voxel_cord[1] = point_nodes[cur_point_global_id].cord.y;
    cur_voxel_cord[2] = point_nodes[cur_point_global_id].cord.x;

    // loop utils list end or max_points reached
    int voxel_point_id = 0;
    while (cur_point_global_id >= 0 &&
           (voxel_point_id < max_points || max_points == 0)) {
      if (reduce_type == reduce_t::MEAN) {
        for (auto j = 0; j < point_dim; ++j) {
          cur_voxel[j] +=
              (points[cur_point_global_id * point_dim + j] - cur_voxel[j]) /
              (voxel_point_id + 1);
        }
      } else if (reduce_type == reduce_t::NEAREST) {
        float point_dist = point_nodes[cur_point_global_id].dist;
        float voxel_dist = voxel_point_id == 0
                               ? std::numeric_limits<float>::max()
                               : cur_voxel_dist;
        if (point_dist < voxel_dist) {
          for (auto j = 0; j < point_dim; ++j) {
            cur_voxel[j] = points[cur_point_global_id * point_dim + j];
          }
          cur_voxel_dist = point_dist;
        }
      } else { // NONE, FIRST
        for (auto j = 0; j < point_dim; ++j) {
          cur_voxel[voxel_point_id * point_dim + j] =
              points[cur_point_global_id * point_dim + j];
        }
      }

      // move to next point
      cur_point_global_id = point_nodes[cur_point_global_id].next;
      voxel_point_id++;
    }

    // record how many final point number in voxel
    voxel_points[voxel_id] = voxel_point_id;
  }

  // clip valid_voxel_num on thread 0
  if (threadIdx.x == 0) {
    int real_valid_voxel_num = min(*valid_voxel_num, max_voxels);
    atomicExch(valid_voxel_num, real_valid_voxel_num);
  }
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
  CHECK_INPUT(points);
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
