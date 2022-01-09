#include "iou3d.h"
#include "nms_bev_kernel.h"
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")

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

namespace iou3d {

using nms_bev::BITMAP_ITEM_BITS;
using nms_bev::BITMAP_ITEM_BYTES;
using nms_bev::THREADS_PER_BLOCK;

using nms_bev::BitmapItem;
using nms_bev::nmsIouCalcKernel;
using nms_bev::nmsSelectKernel;

void nms_bev_gpu(const at::Tensor &boxes, const at::Tensor &scores,
                 const float thresh, const int max_out, at::Tensor &selected,
                 at::Tensor &valid_num) {
  // WARN: assume box_data is sorted(usully comes from topk), we do NOT sort

  CHECK_INPUT(boxes, at::ScalarType::Float);
  CHECK_INPUT(scores, at::ScalarType::Float);
  CHECK_INPUT(selected, at::ScalarType::Int);
  CHECK_INPUT(valid_num, at::ScalarType::Int);
  at::cuda::CUDAGuard device_guard(boxes.device());

  const int box_num = boxes.size(0);
  const int col_blocks = CeilDiv(box_num, BITMAP_ITEM_BITS);

  at::Tensor bitmap =
      boxes.new_zeros({box_num * col_blocks * BITMAP_ITEM_BYTES},
                      boxes.options().dtype(at::ScalarType::Char));
  at::Tensor flag =
      boxes.new_zeros({col_blocks * BITMAP_ITEM_BYTES},
                      boxes.options().dtype(at::ScalarType::Char));

  // kernel to calc relation between boxes
  int grid_dim = CeilDiv(box_num, THREADS_PER_BLOCK);
  nmsIouCalcKernel<<<dim3(grid_dim, grid_dim), THREADS_PER_BLOCK, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
      boxes.data_ptr<float>(), (BitmapItem *)bitmap.data_ptr(), box_num,
      col_blocks, thresh);

  // kernel to select valid boxes, NOTE: use only ONE block
  nmsSelectKernel<<<1, col_blocks, 0, at::cuda::getCurrentCUDAStream()>>>(
      (BitmapItem *)bitmap.data_ptr(), box_num, col_blocks, max_out,
      (BitmapItem *)flag.data_ptr(), selected.data_ptr<int>(),
      valid_num.data_ptr<int>());
}

} // namespace iou3d
