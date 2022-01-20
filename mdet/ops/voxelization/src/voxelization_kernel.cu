#include "voxelization_kernel.h"
#include <limits>
#include <stdio.h>

namespace voxelization {

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

    if (reduce_type == reduce_t::MEAN) {
      for (auto j = 0; j < point_dim; ++j) {
        cur_voxel[j] = 0;
      }
    }

    // loop utils list end or max_points reached
    int voxel_point_id = 0;
    int valid_points = 0;
    while ((cur_point_global_id >= 0 || reduce_type == NONE) &&
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
              cur_point_global_id >= 0
                  ? points[cur_point_global_id * point_dim + j]
                  : 0;
        }
      }

      // move to next point
      ++voxel_point_id;
      if (cur_point_global_id >= 0) {
        cur_point_global_id = point_nodes[cur_point_global_id].next;
        ++valid_points;
      }
    }

    // record how many final point number in voxel
    voxel_points[voxel_id] = min(valid_points, max_points);
  }

  // clip valid_voxel_num on thread 0
  if (threadIdx.x == 0) {
    int real_valid_voxel_num = min(*valid_voxel_num, max_voxels);
    atomicExch(valid_voxel_num, real_valid_voxel_num);
  }
}

} // namespace voxelization
