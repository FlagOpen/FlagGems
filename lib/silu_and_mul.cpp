#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <tuple>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace {
  std::tuple<at::Tensor, at::Tensor, at::ScalarType> prepare_inputs(const at::Tensor &x_,
                                                                    const at::Tensor &y_) {
    auto tensors = torch::broadcast_tensors({x_, y_});
    at::Tensor x = tensors[0].contiguous();
    at::Tensor y = tensors[1].contiguous();
    at::ScalarType out_dtype = at::promote_types(x.scalar_type(), y.scalar_type());
    if (x.scalar_type() != out_dtype) {
      x = x.to(out_dtype);
    }
    if (y.scalar_type() != out_dtype) {
      y = y.to(out_dtype);
    }
    return {x, y, out_dtype};
  }

  void launch_silu_and_mul(const at::Tensor &x,
                           const at::Tensor &y,
                           at::Tensor &out,
                           int64_t block_size = 1024,
                           int num_warps = 4,
                           int num_stages = 1) {
    TORCH_CHECK(x.is_contiguous(), "silu_and_mul: x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "silu_and_mul: y must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "silu_and_mul: out must be contiguous");

    const TritonJITFunction &kernel =
        TritonJITFunction::get_instance(std::string(utils::get_triton_src_path() / "silu_and_mul.py"),
                                        "silu_and_mul_kernel");

    int64_t n = out.numel();
    const unsigned int grid_x = static_cast<unsigned int>((n + block_size - 1) / block_size);

    c10::DeviceGuard guard(out.device());
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());

    kernel(raw_stream, grid_x, 1, 1, num_warps, num_stages, x, y, out, n, block_size);
  }

}  // namespace

at::Tensor silu_and_mul(const at::Tensor &x_, const at::Tensor &y_) {
  auto [x, y, out_dtype] = prepare_inputs(x_, y_);
  at::Tensor out = at::empty_like(x, x.options().dtype(out_dtype));
  launch_silu_and_mul(x, y, out);
  return out;
}

at::Tensor &silu_and_mul_out(at::Tensor &out, const at::Tensor &x_, const at::Tensor &y_) {
  auto [x, y, out_dtype] = prepare_inputs(x_, y_);
  TORCH_CHECK(out.sizes() == x.sizes(),
              "silu_and_mul_out: output tensor must match broadcasted input shape, got ",
              out.sizes(),
              " vs ",
              x.sizes());
  TORCH_CHECK(out.scalar_type() == out_dtype,
              "silu_and_mul_out: output tensor dtype ",
              out.scalar_type(),
              " does not match promoted dtype ",
              out_dtype);
  at::Tensor out_contig;
  if (out.is_contiguous()) {
    out_contig = out;
  } else {
    out_contig = out.contiguous();
  }
  launch_silu_and_mul(x, y, out_contig);
  if (!out.is_contiguous()) {
    out.copy_(out_contig);
  }
  return out;
}

}  // namespace flag_gems
