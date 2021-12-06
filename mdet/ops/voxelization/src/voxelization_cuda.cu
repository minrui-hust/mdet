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

constexpr size_t NDim = 3;

template <typename T_int> inline T_int CeilDiv(T_int a, T_int b) {
  return (a + b - 1) / b;
}

// list all the point in a voxel
struct __align__(4) ListNode {
  int next;  // next point id
  int3 cord; // voxel coord of this point
};

__global__ void ScatterPointToVoxel( // clang-format off
                                     // input
                                     const float* __restrict__ points,  // input points data, point_num*point_dim
                                     int point_num,  // point number,
                                     int point_dim,  // point dimension,
                                     float range_min_x, float range_min_y, float range_min_z, // range
                                     float voxel_size_x, float voxel_size_y, float voxel_size_z, // voxel size
                                     int grid_size_x, int grid_size_y, int grid_size_z, // grid size
                                     int max_voxels, // max output voxels

                                     // output
                                     int *voxel_list_head_id, // store the voxel list head id, indexed by voxel hash id, size: grid_size[0]*grid_size[1]*grid_size[2]
                                     ListNode *point_nodes, // list node of each point, size: point_num
                                     int *output_voxel_hash_id, // out voxel hash id, size: max_voxels
                                     int *valid_voxel_num // output valid voxel number, 1
                                    ){ // clang-format on
  CUDA_1D_KERNEL_LOOP(i, point_num) {
    // get current point and cooresponding list node
    const float *cur_point = points + i * point_dim;
    ListNode *cur_node = point_nodes + i;

    // clang-format off
        // calc which voxel this point scatter into
        auto& cord = cur_node->cord;
        cord.x = floor((cur_point[0] - range_min_x) / voxel_size_x);
        cord.y = floor((cur_point[1] - range_min_y) / voxel_size_y);
        cord.z = floor((cur_point[2] - range_min_z) / voxel_size_z);
    // clang-format on

    // clang-format off
        // drop point out of range
        if (cord.x < 0 || cord.x >= grid_size_x ||
            cord.y < 0 || cord.y >= grid_size_y || 
            cord.z < 0 || cord.z >= grid_size_z ) continue;
    // clang-format on

    // clang-format off
        int hash_id = grid_size_x * grid_size_y * cord.z+ 
                                    grid_size_x * cord.y+ 
                                                  cord.x;
    // clang-format on

    // data in voxel_head_id_addr will be accessed by different thread,
    // protection required
    int *voxel_head_id_addr = voxel_list_head_id + hash_id;

    // atomic set the value in voxel_head_id_addr to be current point index
    // which is 'i'
    int pre_head_id = atomicExch(voxel_head_id_addr, i);

    // point to previous point
    cur_node->next = pre_head_id;

    // check if we should make a new voxel
    // NOTE: valid_voxel_num may be larger than max_voxels
    if (-1 == pre_head_id) {
      int pre_valid_voxel_num = atomicAdd(valid_voxel_num, 1);
      if (pre_valid_voxel_num < max_voxels) {
        output_voxel_hash_id[pre_valid_voxel_num] = hash_id;
      }
    }
  }
}

__global__ void GatherVoxelFromPoint( // clang-format off
                                      // input
                                      const float* __restrict__ points,
                                      int point_num, // point number
                                      int point_dim, // point dimension
                                      const int * __restrict__ voxel_list_head_id, // store the voxel list head id, indexed by voxel hash id, size: grid_size[0]*grid_size[1]*grid_size[2]
                                      const ListNode * __restrict__ point_nodes, // list node of each point, size: point_num
                                      const int * __restrict__ output_voxel_hash_id, // out voxel hash id, size: max_voxels
                                      int * __restrict__ valid_voxel_num, // valid voxel num addr
                                      int max_voxels, // max output voxels
                                      int max_points, // max num of points in a voxel

                                      // output
                                      float *voxels,
                                      int* voxel_cords,
                                      int* voxel_points
                                    ) { // clang-format on
  CUDA_1D_KERNEL_LOOP(i, min(*valid_voxel_num, max_voxels)) {
    const auto &voxel_id = i;
    float *cur_voxel = voxels + voxel_id * max_points * point_dim;
    int *cur_voxel_cord = voxel_cords + voxel_id * 3;

    // the list head
    int cur_point_global_id = voxel_list_head_id[output_voxel_hash_id[i]];

    // NOTE: in reverse order
    cur_voxel_cord[0] = point_nodes[cur_point_global_id].cord.z;
    cur_voxel_cord[1] = point_nodes[cur_point_global_id].cord.y;
    cur_voxel_cord[2] = point_nodes[cur_point_global_id].cord.x;

    // loop utils list end or max_points reached
    int voxel_point_id = 0;
    while (cur_point_global_id >= 0 && voxel_point_id < max_points) {
      // copy point's coors into voxel
      for (auto j = 0; j < point_dim; ++j) {
        cur_voxel[voxel_point_id * point_dim + j] =
            points[cur_point_global_id * point_dim + j];
      }

      // move to next point
      cur_point_global_id = point_nodes[cur_point_global_id].next;
      voxel_point_id++;
    }

    // record how final point number in voxel
    voxel_points[i] = voxel_point_id;

    // clip valid_voxel_num on thread 0
    if (i == 0) {
      int real_valid_voxel_num = min(*valid_voxel_num, max_voxels);
      atomicExch(valid_voxel_num, real_valid_voxel_num);
    }
  }
}

// clang-format off
void VoxelizationAsync( 
                       // input
                       const float* points,  // input points data, point_num*point_dim
                       int point_num,  // point number,
                       int point_dim,  // point dimension,
                       std::vector<float> point_range, // point cloud range
                       std::vector<float> voxel_size, // voxel size
                       std::vector<int> grid_size, // grid size
                       int max_voxels, // max output voxels
                       int max_points, // max points in an voxel

                       // tmp
                       int* voxel_list_head_id, // store the voxel list head id, indexed by voxel hash id, size: grid_size[0]*grid_size[1]*grid_size[2]
                       ListNode* point_nodes, // list node of each point, size: point_num
                       int* output_voxel_hash_id, // out voxel hash id, size: max_voxels

                       // output
                       float* voxels, // output voxels
                       int* voxel_cords, // voxel coordinates
                       int* voxel_points, // how many points in each voxel
                       int* valid_voxel_num, // output valid voxel number, 1

                       // cuda stream
                       cudaStream_t stream
                      ){ // clang-format on
  // cuda kernel grid and block size
  int block, grid;

  // clang-format off
    cudaOccupancyMaxPotentialBlockSize(&grid, &block, ScatterPointToVoxel, 0, point_num);
    grid= std::min(grid, CeilDiv(point_num, block));
    ScatterPointToVoxel<<<grid, block, 0, stream>>>(
        points, point_num, point_dim, 
        point_range[0], point_range[1], point_range[2], 
        voxel_size[0], voxel_size[1], voxel_size[2], 
        grid_size[0], grid_size[1], grid_size[2], 
        max_voxels,
        voxel_list_head_id,
        point_nodes,
        output_voxel_hash_id,
        valid_voxel_num
    );
  // clang-format on

  // clang-format off
    cudaOccupancyMaxPotentialBlockSize(&grid, &block, GatherVoxelFromPoint, 0, max_voxels);
    grid = std::min(grid, CeilDiv(max_voxels, block));
    GatherVoxelFromPoint<<<grid, block, 0, stream>>>(
                                        points, point_num, point_dim, // points
                                        voxel_list_head_id, // store the voxel list head id, indexed by voxel hash id, size: grid_size[0]*grid_size[1]*grid_size[2]
                                        point_nodes, // list node of each point, size: point_num
                                        output_voxel_hash_id, // out voxel hash id, size: max_voxels
                                        valid_voxel_num, // valid voxel num addr
                                        max_voxels, // max output voxels
                                        max_points, // max num of points in a voxel
                                        voxels, // output voxel data
                                        voxel_cords, // output voxel cords
                                        voxel_points
    );
  // clang-format on
}

} // namespace

namespace voxelization {

// clang-format off
void voxelize_gpu(const at::Tensor &points,
                 const std::vector<float> &point_range,
                 const std::vector<float> &voxel_size,
                 const int max_points, 
                 const int max_voxels,
                 const reduce_t reduce_type,
                 at::Tensor &voxels, 
                 at::Tensor &coords,
                 at::Tensor &point_num,
                 at::Tensor &voxel_num){
  // clang-format on
  CHECK_INPUT(points);
  at::cuda::CUDAGuard device_guard(points.device());

  // prepare temporary
  std::vector<int> voxel_reso(NDim);
  for (auto i = 0u; i < NDim; ++i) {
    voxel_reso[i] =
        round((point_range[NDim + i] - point_range[i]) / voxel_size[i]);
  }

  at::Tensor voxel_list_head_id = -at::ones(
      {voxel_reso[0], voxel_reso[1], voxel_reso[2]}, coords.options());

  at::Tensor points_node =
      at::empty({points.size(0) * (int)sizeof(ListNode)},
                coords.options().dtype(at::ScalarType::Byte));

  at::Tensor voxel_hash_id = at::empty({max_voxels}, coords.options());

  // clang-format off
  VoxelizationAsync(points.contiguous().data_ptr<float>(), 
                    points.size(0),
                    points.size(1),
                    point_range,
                    voxel_size,
                    voxel_reso,
                    max_voxels,
                    max_points,
                    voxel_list_head_id.contiguous().data_ptr<int>(),
                    (ListNode*)points_node.contiguous().data_ptr<uint8_t>(),
                    voxel_hash_id.contiguous().data_ptr<int>(),
                    voxels.contiguous().data_ptr<float>(),
                    coords.contiguous().data_ptr<int>(),
                    point_num.contiguous().data_ptr<int>(),
                    voxel_num.contiguous().data_ptr<int>(),
                    at::cuda::getCurrentCUDAStream()
                    );
  // clang-format on

  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace voxelization
