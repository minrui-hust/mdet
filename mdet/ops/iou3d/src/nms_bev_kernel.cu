#include "geometry2d.h"
#include "nms_bev_kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

namespace nms_bev{
using geometry2d::Box;
using geometry2d::EPS;

__device__ inline float iou_bev(const Box &box_a, const Box box_b) {
  float sa = box_a.area();
  float sb = box_b.area();
  float s_overlap = box_a.overlap(box_b);
  return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

__global__ void nmsIouCalcKernel(const float *box_data, BitmapItem *bitmap,
                                 const int box_num, const int col_blocks,
                                 const float thresh) {
  // params: box_data (N, 6) [center.x, center.y, extend.x, extend.y, cos_alpha,
  // sin_alpha]

  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // we only need to calc the top-right corner of the bitmap matrix
  if (row_start > col_start) {
    return;
  }

  const int row_size =
      min(box_num - row_start * THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  const int col_size =
      min(box_num - col_start * THREADS_PER_BLOCK, THREADS_PER_BLOCK);

  __shared__ float block_boxes_data[THREADS_PER_BLOCK * Box::Dim];
  Box *block_boxes = reinterpret_cast<Box *>(block_boxes_data);

  const Box *boxes = reinterpret_cast<const Box *>(box_data);

  if (threadIdx.x < (unsigned)col_size) {
    block_boxes[threadIdx.x] =
        boxes[THREADS_PER_BLOCK * col_start + threadIdx.x];
  }

  __syncthreads();

  if (threadIdx.x < (unsigned)row_size) {
    const int cur_box_idx = THREADS_PER_BLOCK * row_start + threadIdx.x;
    const Box &cur_box = boxes[cur_box_idx];

    int i = 0;
    BitmapItem t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }

    for (i = start; i < col_size; i++) {
      if (iou_bev(cur_box, block_boxes[i]) > thresh) {
        t |= 1ULL << i;
      }
    }

    bitmap[cur_box_idx * col_blocks + col_start] = t;
  }
}

__global__ void nmsSelectKernel(const BitmapItem *bitmap, const int box_num,
                                const int col_blocks, const int max_out,
                                BitmapItem *flag, int *selected_id,
                                int *selected_num) {
  int thread_num = blockDim.x * gridDim.x;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  for (auto box_id = 0; box_id < box_num && (*selected_num) < max_out;
       ++box_id) {
    int block_id = box_id / BITMAP_ITEM_BITS;
    int offset = box_id % BITMAP_ITEM_BITS;
    if (!(flag[block_id] & (1ULL << offset))) {
      // if we get here, mean box_id is an valid box, so we have to do two
      // things:
      // 1. update selected_id and selected_num only on thread 0
      // 2. update the flag concurrently on all thread

      // 1. update selected_id and selected_num only on thread 0,
      //    since we only update on thread 0, no atomic operation is needed
      if (thread_id == 0) {
        selected_id[(*selected_num)++] = box_id;
      }

      // 2. update the flag concurrently on all thread
      for (int col = thread_id; col < col_blocks; col += thread_num) {
        flag[col] |= bitmap[box_id * col_blocks + col];
      }
    }

    // before we move to next iteration,
    // sync is needed to make flags update on different threads seen by all
    // threads
    __syncthreads();
  }
}

} // namespace iou3d
