#pragma once
#include <cuda_runtime.h> // this is required for header to be used in host cpp code

namespace nms_bev{

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
} // namespace iou3d
