#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace pointwise_dynamic {

// 构造函数
// src/flag_gems/utils/pointwise_dynamic.py:prepare_args
/*
args = tuple(
(
StridedBuffer(
  item,
  task_shape,
  broadcasted_stride(item.shape, item.stride(), task_shape),
)
if schema.is_tensor(i)
else item
)
for i, item in enumerate(args)
)
kwargs = {
    k: StridedBuffer(
        item,
        task_shape,
        broadcasted_stride(item.shape, item.stride(), task_shape),
    )
    for k, item in kwargs.items()
}
*/
pointwise_dynamic::StridedBuffer

    Shape
    broadcast(const Shape& s1, const Shape& s2) {
  if (s1.empty()) {
    return s2;
  }
  if (s2.empty()) {
    return s1;
  }

  const Shape* _s1 = &s1;
  const Shape* _s2 = &s2;

  if (_s1->size() < _s2->size()) {
    std::swap(_s1, _s2);
  }

  size_t r1 = _s1->size();
  size_t r2 = _s2->size();
  size_t d = r1 - r2;

  Shape s = *_s1;

  for (size_t i = 0; i < r2; ++i) {
    if ((*_s1)[d + i] == 1) {
      s[d + i] = (*_s2)[i];
    } else if ((*_s2)[i] == 1) {
      s[d + i] = (*_s1)[d + i];
    } else if ((*_s2)[i] == (*_s1)[d + i]) {
      s[d + i] = (*_s2)[i];
    } else {
      std::string msg = "Unbroadcastable shapes: (";
      for (size_t j = 0; j < s1.size(); ++j) msg += std::to_string(s1[j]) + (j < s1.size() - 1 ? ", " : "");
      msg += ") and (";
      for (size_t j = 0; j < s2.size(); ++j) msg += std::to_string(s2[j]) + (j < s2.size() - 1 ? ", " : "");
      msg += ")";
      throw std::invalid_argument(msg);
    }
  }

  return s;
}

template <typename Iterable>
Shape broadcast_shapes(const Iterable& shapes) {
  if (std::empty(shapes)) {
    return {};
  }

  auto it = std::begin(shapes);
  Shape result_shape = *it;
  ++it;

  for (; it != std::end(shapes); ++it) {
    result_shape = broadcast(result_shape, *it);
  }

  return result_shape;
}
};  // namespace pointwise_dynamic

namespace flag_gems {
using namespace triton_jit;
int64_t cdiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}
at::Tensor add_tensor(const at::Tensor& a_, const at::Tensor& b_) {
  // TODO: parse tensor meta info
  std::vector<void*> kernel_params;
  // 2 input
  kernel_params.push(a_);
  kernel_params.push(b_);
  // 1 output
  at::Tensor out = at::empty(a.sizes(), a.options());
  kernel_params.push(&out);
  // if input和output都连续
  // 或者stride相同和第一个tensor torch.ops.aten.is_non_overlapping_and_dense
  // 但是后者不是都连续，为什么stride=1，如连续的ab转置，我们可以忽略它的stride，只计算element wise就行
  // 但返回的时候，是不是要somehow拿回它的stride，不过这可能是python端里的问题
  std::vector<at::Tensor> tensors = {a_, b_, out};
  // WrapperGenerator: gen_kernel_launch
  // KernelGenerator:
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
    kernel_params.push(fast_path_stride_order_ptr);
    kernel_params.push(stride_ptr);
    kernel_params.push(fast_path_stride_order_ptr);
    // stride for output
    kernel_params.push(stride_ptr);

    // task_space -> shape_args... shape = out0.shape
    // use fast path需要考虑shape吗
    // prepare args里设置 task_shape = (tensors[0].numel(),)
    kernel_params.push(task_shape_ptr);
    // num_tasks -> num_tasks = out0.numel()
    kernel_params.push(task_shape_ptr);
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
  // # tile size & tiles_per_cta, gsl style
  // tile_sizes = heuristics_for_tile_size(512, *shape)
  int64_t tile_sizes = 1024;
  int64_t num_tiles = cdiv(task_shape, tile_sizes);  // aka num blocks
  // num_ctas = min(65536, num_tiles)
  int64_t num_ctas = std::min(65536, num_tiles);
  // tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
  int64_t tiles_per_cta = cdiv(num_tiles, num_ctas);
  // one_tile_per_cta = tiles_per_cta==1
  bool one_tile_per_cta = (tiles_per_cta == 1);
  // get function
  std::array<bool, 2> is_tensor;
  checkIfScalar(scalar_tensor, vector_tensor, is_tensor);
  const TritonKernel kernel;
  if (is_tensor[0] && is_tensor[1]) {
    &f = TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "add.py"),
                                        "add_func");
  } else if (is_tensor[0] && !is_tensor[1]) {
    &f = TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "add.py"),
                                        "add_func_tensor_scalar");
  } else if (!is_tensor[0] && is_tensor[1]) {
    &f = TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "add.py"),
                                        "add_func_scalar_tensor");
  } else {
    return a_ + b_;
  }
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  const int num_warps = 8;
  const int num_stages = 1;
  f(stream, num_tiles, 1, 1, num_warps, num_stages, kernel_params);
}

};  // namespace flag_gems
