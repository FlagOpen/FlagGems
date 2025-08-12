#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "c10/util/Logging.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;
using Shape = c10::IntArrayRef;
using Stride = c10::IntArrayRef;
at::Tensor add_tensor(const at::Tensor& a_, const at::Tensor& b_) {
  // TODO: parse tensor meta info
  // LOG(INFO)<< fmt::format("add tensor");
  std::cout << "add tensor";
  std::vector<void*> kernel_params;
  // 2 input
  void* a_ptr = a_.data_ptr();
  void* b_ptr = b_.data_ptr();
  kernel_params.push_back(&a_ptr);
  kernel_params.push_back(&b_ptr);
  int64_t val0 = 1;
  kernel_params.push_back(&val0);

  // calculate task_space
  std::vector<pointwise_dynamic::ShapeR> shapes;
  shapes.push_back(a_.sizes());
  shapes.push_back(b_.sizes());
  pointwise_dynamic::ShapeW task_space = pointwise_dynamic::broadcast_shapes(shapes);
  int ndim = task_space.size();
  // prepare output with size of task_space
  at::Tensor out = at::empty(task_space);
  kernel_params.push_back(&out);
  std::vector<at::Tensor> tensors = {a_, b_, out};
  int task_shape;
  if (pointwise_dynamic::use_fast_path(tensors)) {
    std::cout << "use fast path";
    task_shape = tensors[0].numel();
    void* task_shape_ptr = &task_shape;
    int stride = 1;
    void* stride_ptr = &stride;
    int ndim = 1;
    int fast_path_stride_order = 0;
    void* fast_path_stride_order_ptr = &fast_path_stride_order;
    // push args
    // stride for input
    kernel_params.push_back(stride_ptr);
    kernel_params.push_back(fast_path_stride_order_ptr);
    kernel_params.push_back(stride_ptr);
    kernel_params.push_back(fast_path_stride_order_ptr);
    // stride for output
    kernel_params.push_back(stride_ptr);

    // task_space -> shape_args... shape = out0.shape
    kernel_params.push_back(task_shape_ptr);
    // num_tasks -> num_tasks = out0.numel()
    kernel_params.push_back(task_shape_ptr);
  } else {
    std::cout << "else";
    // broadcast tensor
    // ndim = len(task_shape)
    // shapes = tuple(item.shape for item in in_tensors)
    // task_shape = broadcast_shapes(shapes)
    // c10::IntArrayRef vs at::DimVector

    // broad tensor and warp with StridedBuffer
    // TODO：确定copy机制是否高效
    pointwise_dynamic::StridedBuffer a = pointwise_dynamic::StridedBuffer(
        a_,
        task_shape,
        pointwise_dynamic::broadcasted_stride(a_.sizes(), a_.strides(), task_shape));
    pointwise_dynamic::StridedBuffer b = pointwise_dynamic::StridedBuffer(
        b_,
        task_shape,
        pointwise_dynamic::broadcasted_stride(b_.sizes(), b_.strides(), task_shape));

    // input stride
    const c10::IntArrayRef a_strides = a.strides();
    for (int i = 0; i < ndim; i++) {
      kernel_params.push_back(const_cast<long*>(&a_strides[i]));
    }
    if (ndim >= 2) {
      const pointwise_dynamic::StrideW a_strides_vec(a_strides.begin(), a_strides.end());
      std::vector<int64_t> order_vec = pointwise_dynamic::stride_order(a_strides_vec);
      for (int i = 0; i < ndim; i++) {
        long order_val = order_vec[i];
        kernel_params.push_back(const_cast<long*>(&order_val));
      }
    } else {
      pointwise_dynamic::StrideW zero_stride(1, 0);
      void* zero_stride_ptr = zero_stride.data();
      kernel_params.push_back(&zero_stride_ptr);
    }

    const c10::IntArrayRef b_strides = b.strides();
    for (int i = 0; i < ndim; i++) {
      kernel_params.push_back(const_cast<long*>(&b_strides[i]));
    }
    if (ndim >= 2) {
      const pointwise_dynamic::StrideW b_strides_vec(b_strides.begin(), b_strides.end());
      std::vector<int64_t> order_vec = pointwise_dynamic::stride_order(b_strides_vec);
      for (int i = 0; i < ndim; i++) {
        long order_val = order_vec[i];
        kernel_params.push_back(const_cast<long*>(&order_val));
      }
    } else {
      pointwise_dynamic::StrideW zero_stride(1, 0);
      void* zero_stride_ptr = zero_stride.data();
      kernel_params.push_back(&zero_stride_ptr);
    }
    // output stride
    // TODO：封装 push 1d tensor metadata的函数
    const c10::IntArrayRef output_strides = out.strides();
    for (int i = 0; i < ndim; i++) {
      kernel_params.push_back(const_cast<long*>(&output_strides[i]));
    }
    if (ndim >= 2) {
      const pointwise_dynamic::StrideW output_strides_vec(output_strides.begin(), output_strides.end());
      std::vector<int64_t> order_vec = pointwise_dynamic::stride_order(output_strides_vec);
      for (int i = 0; i < ndim; i++) {
        long order_val = order_vec[i];
        kernel_params.push_back(const_cast<long*>(&order_val));
      }
    } else {
      pointwise_dynamic::StrideW zero_stride(1, 0);
      void* zero_stride_ptr = zero_stride.data();
      kernel_params.push_back(&zero_stride_ptr);
    }

    // task space
    for (int i = 0; i < ndim; i++) {
      int64_t si = task_space[i];
      kernel_params.push_back(const_cast<int64_t*>(&si));
    }
    // num_task out的
    int64_t num_task = out.numel();
    kernel_params.push_back(const_cast<int64_t*>(&num_task));
  }
  void* global_scratch = nullptr;
  kernel_params.push_back(&global_scratch);
  // # tile size & tiles_per_cta, gsl style
  // tile_sizes = heuristics_for_tile_size(512, *shape)
  int64_t tile_sizes = 1024;
  int64_t num_tiles = utils::cdiv(task_shape, tile_sizes);  // aka num blocks
  // num_ctas = min(65536, num_tiles)
  int64_t num_ctas = std::min(static_cast<int64_t>(65536), num_tiles);
  // tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
  int64_t tiles_per_cta = utils::cdiv(num_tiles, num_ctas);
  // one_tile_per_cta = tiles_per_cta==1
  bool one_tile_per_cta = (tiles_per_cta == 1);
  // get function
  std::array<bool, 2> is_tensor;
  pointwise_dynamic::checkIfScalar(a_, b_, is_tensor);
  std::optional<TritonJITFunction> f;
  // TODO: code gen in c++
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
  f->launch_with_raw_args(raw_stream,
                          num_tiles,
                          1,
                          1,
                          num_warps,
                          num_stages,
                          signature,
                          kernel_params.data());
  return out;
}

};  // namespace flag_gems
