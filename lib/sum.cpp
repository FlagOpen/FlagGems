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

namespace flag_gems {
using namespace triton_jit;
// sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
at::Tensor sum_dim(const at::Tensor &self, ::std::optional<at::ScalarType> dtype) {
}

// signature
// sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType?
// dtype=None) -> Tensor
at::Tensor sum_dim(const at::Tensor &self,
                   at::OptionalIntArrayRef dim,
                   bool keepdim,
                   ::std::optional<at::ScalarType> dtype) {
  at::DimVector dims_ = at::native::make_dim_vector(dim, self.dim());
  at::maybe_wrap_dims(dims_, self.dim());
  at::DimVector shape = at::meta::get_reduction_shape(self, dims_, keepdim, false);
  c10::ScalarType out_dtype = at::native::get_dtype_from_self(self, dtype, true);
  at::Tensor out = at::empty(shape, self.options());

  auto [permuted_self, non_reduction_size, reduction_size] = utils::permute_reduction_axes_right(self, dims_);
  permuted_self = permuted_self.contiguous();

  /* signature to remind yourself
  def sum_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
  ):
  */
  const TritonJITFunction &f =
      TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "sum.py"),
                                     "sum_kernel");

  // add utility to build this automatically
  int64_t tile_m = 4;
  int64_t tile_n = 512;
  const int num_warps = 8;
  const int num_stages = 2;
  const unsigned int num_blocks = (non_reduction_size + tile_m - 1) / tile_m;

  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(raw_stream,
    num_blocks,
    1,
    1,
    num_warps,
    num_stages,
    permuted_self,
    out,
    non_reduction_size,
    reduction_size,
    tile_m,
    tile_n,
    num_stages);
  return out;
}

}  // namespace flag_gems
