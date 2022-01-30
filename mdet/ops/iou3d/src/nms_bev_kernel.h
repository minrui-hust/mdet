#pragma once
#include <cuda_runtime.h> // this is required for header to be used in host cpp code

namespace nms_bev {

using BitmapItem = unsigned long long;
constexpr int BITMAP_ITEM_BYTES = sizeof(BitmapItem);
constexpr int BITMAP_ITEM_BITS = BITMAP_ITEM_BYTES * 8;
constexpr int THREADS_PER_BLOCK = BITMAP_ITEM_BITS;

__global__ void nmsIouCalcKernel(const float *box_data, BitmapItem *bitmap,
                                 const int box_num, const int col_blocks,
                                 const float thresh);

__global__ void nmsSelectKernel(const BitmapItem *bitmap, const int box_num,
                                const int col_blocks, const int max_out,
                                BitmapItem *flag, int *selected_id,
                                int *selected_num);

__global__ void iouCalcKernel(const float *pbox_data, const float *qbox_data,
                              const int box_num, float *iou_data);

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                   \
       i += blockDim.x * gridDim.x)

} // namespace nms_bev
