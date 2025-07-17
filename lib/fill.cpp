#include "c10/cuda/CUDAStream.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

// ----------- Scalar fill: create a new tensor and fill it with a scalar value -----------
at::Tensor fill_scalar(const at::Tensor& self, const c10::Scalar& value) {
  at::Tensor out = at::empty_like(self);
  int64_t numel = out.numel();
  if (numel == 0) return out;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  TritonJITFunction fill_kernel =
      TritonJITFunction::getInstance((utils::get_triton_src_path() / "fill.py").string(),
                                     "fill_scalar_kernel");

  c10::DeviceGuard guard(out.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  fill_kernel(stream, grid_x, 1, 1, 4, 0, out, value, numel, BLOCK_SIZE);

  return out;
}

// ----------- Tensor fill: create a new tensor and fill it with a 0-dimensional tensor value -----------
at::Tensor fill_tensor(const at::Tensor& self, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_tensor only supports 0-dim value tensor");
  at::Tensor out = at::empty_like(self);
  int64_t numel = out.numel();
  if (numel == 0) return out;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  TritonJITFunction fill_kernel =
      TritonJITFunction::getInstance((utils::get_triton_src_path() / "fill.py").string(),
                                     "fill_tensor_kernel");

  c10::DeviceGuard guard(out.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  fill_kernel(stream, grid_x, 1, 1, 4, 0, out, value, numel, BLOCK_SIZE);

  return out;
}

// ----------- Scalar fill inplace -----------
at::Tensor& fill_scalar_(at::Tensor& self, const c10::Scalar& value) {
  int64_t numel = self.numel();
  if (numel == 0) return self;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  TritonJITFunction fill_kernel =
      TritonJITFunction::getInstance((utils::get_triton_src_path() / "fill.py").string(),
                                     "fill_scalar_kernel");

  c10::DeviceGuard guard(self.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  fill_kernel(stream, grid_x, 1, 1, 4, 0, self, value, numel, BLOCK_SIZE);

  return self;
}

// ----------- Tensor fill inplace -----------
at::Tensor& fill_tensor_(at::Tensor& self, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_tensor_ only supports 0-dim value tensor");
  int64_t numel = self.numel();
  if (numel == 0) return self;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  TritonJITFunction fill_kernel =
      TritonJITFunction::getInstance((utils::get_triton_src_path() / "fill.py").string(),
                                     "fill_tensor_kernel");

  c10::DeviceGuard guard(self.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  fill_kernel(stream, grid_x, 1, 1, 4, 0, self, value, numel, BLOCK_SIZE);

  return self;
}

}  // namespace flag_gems
