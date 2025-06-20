#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor mm_tensor(const at::Tensor &mat1, const at::Tensor &mat2) {
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "both the tensors must be 2-D");
  TORCH_CHECK(mat1.dtype() == mat2.dtype(),
              "expected a and b to have the same dtype, but got: ",
              mat1.dtype(),
              " != ",
              mat2.dtype())

  at::IntArrayRef mat1_sizes = mat1.sizes();
  at::IntArrayRef mat2_sizes = mat2.sizes();
  at::Tensor out = at::empty({mat1_sizes[0], mat2_sizes[1]}, mat1.options());

  const TritonJITFunction &f =
      TritonJITFunction::getInstance(std::string(utils::get_triton_src_path() / "mm.py"), "mm_kernel");

  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  unsigned int grid_x = ((mat1_sizes[0] + 127) / 128) * ((mat2_sizes[1] + 127) / 128);
  f(/* CUstream = */ raw_stream,
    /* grid_x = */ grid_x,
    /* grid_y = */ 1,
    /* grid_z = */ 1,
    /* num_warps = */ 4,
    /* num_stages = */ 1,
    mat1,
    mat2,
    out,
    mat1_sizes[0],
    mat2_sizes[1],
    mat1_sizes[1],
    mat1.stride(0),
    mat1.stride(1),
    mat2.stride(0),
    mat2.stride(1),
    out.stride(0),
    out.stride(1),
    /* BLOCK_M = */ 128,
    /* BLOCK_N = */ 128,
    /* BLOCK_K = */ 32,
    /* GROUP_M = */ 4,
    /* SPLIT_K = */ 1);
  return out;
}
}  // namespace flag_gems
