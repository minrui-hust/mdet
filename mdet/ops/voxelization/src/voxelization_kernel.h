#pragma once
#include <cuda_runtime.h>

namespace voxelization {

enum reduce_t {
  NONE = 0,
  MEAN = 1,
  FIRST = 2,
  NEAREST = 3,
};

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
); // clang-format on

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
); // clang-format on

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
); // clang-format on

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
); // clang-format on

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                   \
       i += blockDim.x * gridDim.x)

} // namespace voxelization
