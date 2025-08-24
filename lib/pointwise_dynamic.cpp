#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "c10/util/Logging.h"
#include "pybind11/embed.h"
#include "triton_jit/pointwise_generator.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace py = pybind11;
at::Tensor add_tensor(const at::Tensor& a_, const at::Tensor& b_) {
  pointwise_dynamic::ParamStack stk = pointwise_dynamic::ParamStack();
  int64_t task_shape, ndim;
  int64_t num_ctas;
  int64_t tiles_per_cta;
  int64_t tile_sizes;
  int64_t num_tiles;
  at::Tensor out = at::empty_like(a_);
  std::vector<at::Tensor> tensors = {a_, b_, out};
  const int num_warps = 4;
  const int num_stages = 1;
  if (pointwise_dynamic::use_fast_path(tensors)) {
    task_shape = a_.numel();
    int64_t stride = 1;
    ndim = 1;
    stk.save_stride(stride);
    stk.save_stride(stride);
    stk.save_stride(stride);
    stk.save_task_shape(task_shape);
    stk.save_task_shape(task_shape);
    tile_sizes = num_warps * 32;
    num_tiles = utils::cdiv(task_shape, tile_sizes);
    num_ctas = std::min(static_cast<int64_t>(65536), num_tiles);
    tiles_per_cta = utils::cdiv(num_tiles, num_ctas);
    stk.save_task_partition(tiles_per_cta);
  } else {
    std::runtime_error("NotImplementError");
  }
  stk.save_constexpr(tile_sizes);
  int64_t one_tile_per_cta = (tiles_per_cta == 1);
  stk.save_constexpr(one_tile_per_cta);

  std::array<bool, 2> is_scalar;
  pointwise_dynamic::checkIfScalar(a_, b_, is_scalar);
  std::optional<TritonJITFunction> f;
  auto ans_tuple = gen_add(ndim);
  std::string kernel_name = std::get<0>(ans_tuple);
  std::string file_path = std::get<1>(ans_tuple);
  if (!is_scalar[0] && !is_scalar[1]) {
    f = TritonJITFunction::getInstance(file_path, kernel_name);
  } else if (!is_scalar[0] && is_scalar[1]) {
    std::runtime_error("NotImplementError");
    f = TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "add.py"),
                                       "add_func_tensor_scalar");
  } else if (is_scalar[0] && !is_scalar[1]) {
    std::runtime_error("NotImplementError");
    f = TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "add.py"),
                                       "add_func_scalar_tensor");
  } else {
    return a_ + b_;
  }
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  stk.save_tensor(a_);
  stk.save_tensor(b_);
  stk.save_tensor(out);
  stk.build();
  f->launch_with_raw_args(raw_stream,
                          num_ctas,
                          1,
                          1,
                          num_warps,
                          num_stages,
                          stk.get_signature(),
                          stk.get_params());
  return out;
}

};  // namespace flag_gems
