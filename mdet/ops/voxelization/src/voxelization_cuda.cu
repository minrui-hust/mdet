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
        cord.z < 0 || cord.z >= voxel_reso_z ) continue;
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

__global__ void ScatterPillarPointToVoxel( // clang-format off
  // input
  const float* __restrict__ points, int point_num, int point_dim,  // points,
  ListNode * __restrict__ point_nodes, // node for each point, store tmp info of the point
  const int * __restrict__ map_key_to_pillar_list,
  const int * __restrict__ map_pillar_to_key,
  const int* __restrict__ pillar_num,
  float range_min_z,
  float voxel_size_z,
  int voxel_reso_z,
  int max_voxels,

  // output
  int *map_key_to_voxel_list, // point key to voxel id, init to -1, size: pillar_num*voxel_reso_z
  int *map_voxel_to_key, // voxel to voxel's first point, init to -1, size: max_voxels
  int *valid_voxel_num // output valid voxel number, init to 0
){ // clang-format on
  CUDA_1D_KERNEL_LOOP(i, *pillar_num) {
    const auto &pillar_id = i;

    int point_id = map_key_to_pillar_list[map_pillar_to_key[pillar_id]];
    while (point_id >= 0) {
      ListNode *cur_node = point_nodes + point_id;

      int key = voxel_reso_z * pillar_id + cur_node->cord.z;

      int pre_node_id = atomicExch(map_key_to_voxel_list + key, point_id);

      if (pre_node_id == -1) {
        int voxel_id = atomicAdd(valid_voxel_num, 1);
        if (voxel_id < max_voxels) {
          map_voxel_to_key[voxel_id] = key;
        }
      }

      // move on to next
      point_id = cur_node->next;
      cur_node->next = pre_node_id;
    } // end while
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

// clang-format off
} // namespace

namespace voxelization {

// clang-format off
void voxelize_gpu(const at::Tensor &points,
                 const std::vector<float> &point_range,
                 const std::vector<float> &voxel_size,
                 const std::vector<int32_t> &voxel_reso,
                 const int max_points, 
                 const int max_voxels,
                 const reduce_t reduce_type,
                 at::Tensor &voxels, 
                 at::Tensor &coords,
                 at::Tensor &voxel_points,
                 at::Tensor &voxel_num){
  // clang-format on
  CHECK_INPUT(points);
  at::cuda::CUDAGuard device_guard(points.device());

  int point_num = points.size(0);
  int point_dim = points.size(1);

  // allocate device memory
  bool pillar_only = (voxel_reso[2] <= 1);
  int max_pillars = pillar_only ? max_voxels : point_num;

  int device_data_bytes = 0;

  int point_nodes_bytes = point_num * sizeof(ListNode);
  device_data_bytes += point_nodes_bytes;

  int map_key_to_pillar_list_bytes =
      voxel_reso[0] * voxel_reso[1] * sizeof(int);
  device_data_bytes += map_key_to_pillar_list_bytes;

  int map_pillar_to_key_bytes = max_pillars * sizeof(int);
  device_data_bytes += map_pillar_to_key_bytes;

  int voxel_dists_bytes =
      reduce_type == reduce_t::NEAREST ? max_voxels * sizeof(float) : 0;
  device_data_bytes += voxel_dists_bytes;

  int map_key_to_voxel_list_bytes =
      pillar_only ? 0 : (max_pillars * voxel_reso[2] * sizeof(int));
  device_data_bytes += map_key_to_voxel_list_bytes;

  int map_voxel_to_key_bytes = pillar_only ? 0 : (max_voxels * sizeof(int));
  device_data_bytes += map_voxel_to_key_bytes;

  int pillar_num_bytes = pillar_only ? 0 : sizeof(int);
  device_data_bytes += pillar_num_bytes;

  int8_t *device_data;
  cudaMalloc(&device_data, device_data_bytes);
  // printf("device_data_bytes: %d", device_data_bytes);

  // clang-format off
  int8_t *point_nodes_data            = device_data;
  int8_t *map_key_to_pillar_list_data = point_nodes_data + point_nodes_bytes;
  int8_t *map_pillar_to_key_data      = map_key_to_pillar_list_data + map_key_to_pillar_list_bytes;
  int8_t *voxel_dists_data            = map_pillar_to_key_data + map_pillar_to_key_bytes;
  int8_t *map_key_to_voxel_list_data  = voxel_dists_data + voxel_dists_bytes;
  int8_t *map_voxel_to_key_data       = map_key_to_voxel_list_data + map_key_to_voxel_list_bytes;
  int8_t *pillar_num_data             = map_voxel_to_key_data + map_voxel_to_key_bytes;
  // clang-format on

  // map_key_to_pillar_list -> -1
  at::from_blob(map_key_to_pillar_list_data, {voxel_reso[0], voxel_reso[1]},
                coords.options())
      .fill_(-1);

  // map_key_to_voxel_list -> -1
  if (map_key_to_voxel_list_bytes > 0) {
    at::from_blob(map_key_to_voxel_list_data, {max_pillars, voxel_reso[2]},
                  coords.options())
        .fill_(-1);
  }

  // pillar_num -> 0
  if (pillar_num_bytes > 0) {
    cudaMemsetAsync(pillar_num_data, 0, sizeof(int),
                    at::cuda::getCurrentCUDAStream());
  }

  int block, grid;

  // clang-format off
  // point to pillar
  cudaOccupancyMaxPotentialBlockSize(&grid, &block, ScatterPointToPillar, 0, point_num);
  grid= std::min(grid, CeilDiv(point_num, block));
  ScatterPointToPillar<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      points.data_ptr<float>(), point_num,  point_dim,
      point_range[0], point_range[1], point_range[2],
      voxel_size[0],  voxel_size[1],  voxel_size[2],
      voxel_reso[0],  voxel_reso[1],  voxel_reso[2],
      max_pillars,
      (reduce_type==reduce_t::NEAREST), // whether record distance

      // output
      (ListNode*)point_nodes_data, // companion node of each point, same number as points
      (int *)map_key_to_pillar_list_data, // point key to pillar id hash map, initialized to -1, size: pillar_reso_x*pillar_reso_y
      (int *)map_pillar_to_key_data, // pillar id to head node id of the list of this pillar, size: max_pillars, initialized to -1(if append_to_pillar set to true)
      pillar_only?voxel_num.data_ptr<int>():(int*)pillar_num_data // valid pillar number, init to 0
  );

  // point to voxel if needed
  if(!pillar_only){
      cudaOccupancyMaxPotentialBlockSize(&grid, &block, ScatterPillarPointToVoxel, 0, point_num);
      grid= std::min(grid, CeilDiv(point_num, block));
      ScatterPillarPointToVoxel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          points.data_ptr<float>(), point_num, point_dim,
          (ListNode *) point_nodes_data,
          (int *) map_key_to_pillar_list_data,
          (int *) map_pillar_to_key_data,
          (int* )pillar_num_data,
          point_range[2],
          voxel_size[2],
          voxel_reso[2],
          max_voxels,
          (int *)map_key_to_voxel_list_data,
          (int *)map_voxel_to_key_data,
          voxel_num.data_ptr<int>()
      );
  }else{
    map_key_to_voxel_list_data = map_key_to_pillar_list_data;
    map_voxel_to_key_data = map_pillar_to_key_data;
  }

  // gather voxel/pillar from points
  cudaOccupancyMaxPotentialBlockSize(&grid, &block, GatherVoxelFromPoint, 0, max_voxels);
  grid = std::min(grid, CeilDiv(max_voxels, block));
  GatherVoxelFromPoint<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      points.data_ptr<float>(), point_num, point_dim,
      (ListNode *) point_nodes_data,
      (int *) map_key_to_voxel_list_data, // voxel id to head node id of the list of this voxel, size: max_voxels, filled by ScatterPointToVoxel
      (int *) map_voxel_to_key_data, // voxel id to head node id of the list of this voxel, size: max_voxels, filled by ScatterPointToVoxel
      voxel_num.data_ptr<int>(),
      max_voxels,
      max_points,
      reduce_type,
      (float*) voxel_dists_data,
      voxels.data_ptr<float>(),
      coords.data_ptr<int>(),
      voxel_points.data_ptr<int>()
  );
  // clang-format on

  cudaFree(device_data);
  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace voxelization
