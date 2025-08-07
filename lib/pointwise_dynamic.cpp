#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;
using Shape = std::vector<long>;
using Stride = std::vector<long>;

at::Tensor add_tensor(const at::Tensor& a_, const at::Tensor& b_) {
  // TODO: parse tensor meta info
  std::vector<void*> kernel_params;
  // 2 input
  void* a_ptr = a_.data_ptr();
  void* b_ptr = b_.data_ptr();
  kernel_params.push_back(&a_ptr);
  kernel_params.push_back(&b_ptr);

  // 1 output
  at::Tensor out = at::empty(a_.sizes(), a_.options());
  kernel_params.push_back(&out);
  std::vector<at::Tensor> tensors = {a_, b_, out};
  if (pointwise_dynamic::use_fast_path(tensors)) {
    int task_shape = tensors[0].numel();
    void* task_shape_ptr = &task_shape;
    int stride = 1;
    void* stride_ptr = &stride;
    int ndim = 1;
    int fast_path_stride_order = 0;
    void* fast_path_stride_order_ptr = &fast_path_stride_order
                                            // push args
                                            // stride for input
                                            kernel_params.push(stride_ptr);
    kernel_params.push_back(fast_path_stride_order_ptr);
    kernel_params.push_back(stride_ptr);
    kernel_params.push_back(fast_path_stride_order_ptr);
    // stride for output
    kernel_params.push_back(stride_ptr);

    // task_space -> shape_args... shape = out0.shape
    // use fast path需要考虑shape吗
    // prepare args里设置 task_shape = (tensors[0].numel(),)
    kernel_params.push_back(task_shape_ptr);
    // num_tasks -> num_tasks = out0.numel()
    kernel_params.push_back(task_shape_ptr);
  } else {
    // TODO
    //  stride for input/output
    //  calculate task space
    //  shapes = tuple(item.shape for item in in_tensors)，
    std::vector<Shape> shapes;
    shapes.reserve(2);
    for (const auto& tensor : in_tensors) {
      shapes.push_back(tensor.shape());
    }
    Shape task_shape = broadcast_shapes(shapes);
    int64_t ndim = task_shape.size();
    // task_shape = broadcast_shapes(shapes)
    // get stride, TODO，using ndim as python warpper
    auto a_strides = a_.strides();
    for (int64_t stride : a_strides) {
      kernel_params.push_back(&stride);
    }
    auto b_strides = b_.strides();
    for (int64_t stride : b_strides) {
      kernel_params.push_back(&stride);
    }
    auto out_strides = out.strides();
    for (int64_t stride : out_strides) {
      kernel_params.push_back(&stride);
    }
  }
  void* global_scratch = nullptr;
  kernel_params.push_back(&global_scratch);
  // # tile size & tiles_per_cta, gsl style
  // tile_sizes = heuristics_for_tile_size(512, *shape)
  int64_t tile_sizes = 1024;
  int64_t num_tiles = utils::cdiv(task_shape, tile_sizes);  // aka num blocks
  // num_ctas = min(65536, num_tiles)
  int64_t num_ctas = std::min(65536, num_tiles);
  // tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
  int64_t tiles_per_cta = utils::cdiv(num_tiles, num_ctas);
  // one_tile_per_cta = tiles_per_cta==1
  bool one_tile_per_cta = (tiles_per_cta == 1);
  // get function
  std::array<bool, 2> is_tensor;
  checkIfScalar(scalar_tensor, vector_tensor, is_tensor);
  TritonJITFunction f;
  if (is_tensor[0] && is_tensor[1]) {
    f = TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "add.py"),
                                       "add_func");
  } else if (is_tensor[0] && !is_tensor[1]) {
    f = TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "add.py"),
                                       "add_func_tensor_scalar");
  } else if (!is_tensor[0] && is_tensor[1]) {
    f = TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "add.py"),
                                       "add_func_scalar_tensor");
  } else {
    return a_ + b_;
  }
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  const int num_warps = 8;
  const int num_stages = 1;

  std::string signature = "*fp32:16,*fp32:16,*fp32:16,i64,1024";
  f.launch_with_raw_args(raw_stream, num_tiles, 1, 1, num_warps, num_stages, signature, kernel_params.data());
  return out;
}

};  // namespace flag_gems
