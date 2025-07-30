#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#include <filesystem>
#include "ATen/WrapDimUtils.h"
#include "ATen/native/ReduceOpsUtils.h"
#include "c10/util/DimVector.h"

namespace {
std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(const at::Tensor &tensor,
                                                                      int reduction_axis) {
  int64_t dim = tensor.dim();
  c10::DimVector left_axes, right_axes;
  int64_t non_reduction_size = 1, reduction_size = 1;

  for (int64_t i = 0; i < dim; ++i) {
    if (i == reduction_axis) {
      right_axes.push_back(i);
      reduction_size *= tensor.size(i);
    } else {
      left_axes.push_back(i);
      non_reduction_size *= tensor.size(i);
    }
  }
  c10::DimVector permute_order = left_axes;
  permute_order.insert(permute_order.end(), right_axes.begin(), right_axes.end());

  return {tensor.permute(permute_order), non_reduction_size, reduction_size};
}
int next_power_of_2(int x) {
  return (1 << (32 - __builtin_clz(x - 1)));
}
int cdiv(int a, int b) {
  return (a + b - 1) / b;
}
}  // anonymous namespace

namespace flag_gems {
using namespace triton_jit;
// max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) ->
// (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor, at::Tensor> max_dim_max(const at::Tensor &self,
                                                 int64_t dim,
                                                 bool keepdim,
                                                 const at::Tensor out_value,
                                                 const at::Tensor out_index) {
  auto [permuted_self, non_reduction_size, reduction_size] = permute_reduction_axes_right(self, dim);
  // set_output(out_value,out_index);
  permuted_self = permuted_self.contiguous();
  const TritonJITFunction &f =
      TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "max.py"),
                                     "max_kernel");
  int64_t tile_m = 4;
  int64_t tile_n = 512;
  const int num_warps = 8;
  const int num_stages = 2;
  const unsigned int num_blocks = (non_reduction_size + tile_m - 1) / tile_m;
  /*
  def max_kernel(
      inp,
      out_value,
      out_index,
      M,
      N,
      BLOCK_M: tl.constexpr,
      BLOCK_N: tl.constexpr,
  ):
  */
  c10::DeviceGuard guard(out_value.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  f(stream,
    num_blocks,
    1,
    1,
    num_warps,
    num_stages,
    permuted_self,
    out_value,
    out_index,
    non_reduction_size,
    reduction_size,
    tile_m,
    tile_n);

  return std::make_tuple(out_value, out_index);
}
// max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor, at::Tensor> max_dim(const at::Tensor &self, int64_t dim, bool keepdim) {
  at::DimVector shape = at::meta::get_reduction_shape(self, dim, keepdim, false);
  at::Tensor out_value = at::empty(shape, self.options());
  at::Tensor out_index = at::empty(shape, self.options().dtype(at::kLong));

  auto [permuted_self, non_reduction_size, reduction_size] = permute_reduction_axes_right(self, dim);
  permuted_self = permuted_self.contiguous();
  const TritonJITFunction &f =
      TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "max.py"),
                                     "max_kernel");
  int64_t tile_m = 4;
  int64_t tile_n = 512;
  const int num_warps = 8;
  const int num_stages = 2;
  const unsigned int num_blocks = (non_reduction_size + tile_m - 1) / tile_m;
  /*
  def max_kernel(
      inp,
      out_value,
      out_index,
      M,
      N,
      BLOCK_M: tl.constexpr,
      BLOCK_N: tl.constexpr,
  ):
  */
  c10::DeviceGuard guard(out_value.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  f(stream,
    num_blocks,
    1,
    1,
    num_warps,
    num_stages,
    permuted_self,
    out_value,
    out_index,
    non_reduction_size,
    reduction_size,
    tile_m,
    tile_n);

  return std::make_tuple(out_value, out_index);
}

at::Tensor max(const at::Tensor &self) {
  TORCH_CHECK(self.is_contiguous(), "Input tensor must be contiguous");
  int64_t M = self.numel();
  int64_t block_size = 1 << static_cast<int>(std::ceil(std::log2(std::sqrt(M))));
  int64_t mid_size = (M + block_size - 1) / block_size;
  int64_t block_mid = 1 << static_cast<int>(std::ceil(std::log2(mid_size)));

  at::Tensor mid = torch::empty({mid_size}, self.options());
  at::Tensor out = torch::empty({}, self.options());

  const TritonJITFunction &max_kernel_1 =
      TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "max.py"),
                                     "max_kernel_1");
  const TritonJITFunction &max_kernel_2 =
      TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "max.py"),
                                     "max_kernel_2");
  block_size = next_power_of_2(static_cast<int>(std::ceil(std::sqrt(M))));
  mid_size = cdiv(M, block_size);
  block_mid = next_power_of_2(mid_size);
  const int num_warps = 8;
  const int num_stages = 2;
  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  // def max_kernel_1(inp,mid,M,BLOCK_SIZE: tl.constexpr)
  max_kernel_1(stream, mid_size, 1, 1, num_warps, num_stages, self, mid, M, block_size);
  // def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
  max_kernel_2(stream, 1, 1, 1, num_warps, num_stages, mid, out, mid_size, block_mid);
  return out;
}
}  // namespace flag_gems
