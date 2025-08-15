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

/*
def add_func(
    in0_ptr: tl.tensor, # of tl.pointer_type
    in1_ptr: tl.tensor, # of tl.pointer_type
    out0_ptr: tl.tensor, # of tl.pointer_type
    in0_stride0: int, in0_stride1: int, # strides for in0
    in1_stride0: int, in1_stride1: int, # strides for in1
    out0_stride0: int, out0_stride1: int, # strides for out0
    s0: int, s1: int, # task_space
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
*/

namespace py = pybind11;
at::Tensor add_tensor(const at::Tensor& a_, const at::Tensor& b_) {
  // TODO: parse tensor meta info
  // LOG(INFO)<< fmt::format("add tensor");
  std::string signature;
  std::vector<void*> kernel_params;
  pointwise_dynamic::ParamStack stk = pointwise_dynamic::ParamStack();
  // 2 input
  void* a_ptr = a_.data_ptr();
  void* b_ptr = b_.data_ptr();
  kernel_params.push_back(&a_ptr);
  signature.append("*fp32:16,");
  kernel_params.push_back(&b_ptr);
  signature.append("*fp32:16,");
  // TODO: use fast path没有这个，但
  // int64_t val0 = 1;
  // signature.push("1,");
  // kernel_params.push_back(&val0);
  // general args
  int64_t ndim;
  int64_t num_ctas;
  int64_t tiles_per_cta;
  int64_t tile_sizes;
  at::Tensor out = at::empty_like(a_);
  void* out_ptr = out.data_ptr();
  kernel_params.push_back(&out_ptr);
  signature.append("*fp32:16,");
  std::vector<at::Tensor> tensors = {a_, b_, out};
  int64_t task_shape;
  const int num_warps = 4;  // TODO：pointwise codegen 静态指定
  const int num_stages = 1;
  if (pointwise_dynamic::use_fast_path(tensors)) {
    // prepare output with size of task_space
    std::cout << "use fast path\n";
    task_shape = a_.numel();
    void* task_shape_ptr = &task_shape;
    int64_t stride = 1;
    void* stride_ptr = &stride;
    ndim = 1;
    int64_t fast_path_stride_order = 0;
    void* fast_path_stride_order_ptr = &fast_path_stride_order;
    // push args
    // stride for input
    // kernel_params.push_back(stride_ptr);
    signature.append("i64:1,");
    // kernel_params.push_back(fast_path_stride_order_ptr);
    // kernel_params.push_back(stride_ptr);
    signature.append("i64:1,");
    // kernel_params.push_back(fast_path_stride_order_ptr);
    // stride for output
    // kernel_params.push_back(stride_ptr);
    signature.append("i64:1,");
    stk.save_stride(stride);
    stk.save_stride(stride);
    stk.save_stride(stride);
    // task_space -> shape_args... shape = out0.shape
    kernel_params.push_back(task_shape_ptr);
    signature.append("i64,");
    stk.save_task_shape(task_shape);
    // num_tasks -> num_tasks = out0.numel()
    kernel_params.push_back(task_shape_ptr);
    signature.append("i64,");
    stk.save_task_shape(task_shape);

    int64_t tile_sizes = num_warps * 32;
    int64_t num_tiles = utils::cdiv(task_shape, tile_sizes);  // aka num blocks

    // num_ctas = min(65536, num_tiles)
    num_ctas = std::min(static_cast<int64_t>(65536), num_tiles);
    // tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    tiles_per_cta = utils::cdiv(num_tiles, num_ctas);
    void* tiles_per_cta_ptr = &tiles_per_cta;
    // kernel_params.push_back(tiles_per_cta_ptr);
    signature.append("i64:1,");
    // stk.save_task_partition(tiles_per_cta);
  } else {
    // calculate task_space
    std::vector<pointwise_dynamic::ShapeR> shapes;
    shapes.push_back(a_.sizes());
    shapes.push_back(b_.sizes());
    pointwise_dynamic::ShapeW task_space = pointwise_dynamic::broadcast_shapes(shapes);
    ndim = task_space.size();
    // prepare output with size of task_space
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
    tile_sizes = num_warps * 32;
    int64_t num_tiles = utils::cdiv(task_shape, tile_sizes);  // aka num blocks
    // num_ctas = min(65536, num_tiles)
    /* TODO，处理tiles_per_cta 这件事
      num_ctas = std::min(static_cast<int64_t>(65536), num_tiles);
      // tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
      int64_t tiles_per_cta = utils::cdiv(num_tiles, num_ctas);
      void* tiles_per_cta_ptr = &tiles_per_cta;
      kernel_params.push_back(tiles_per_cta_ptr);
      // num_tasks -> num_tasks = out0.numel()
      kernel_params.push_back(task_shape_ptr);
      // num_task out的
      int64_t num_task = out.numel();
      kernel_params.push_back(const_cast<int64_t*>(&num_task));
    */
  }
  signature.append(std::to_string(tile_sizes));
  signature.append(",");
  stk.save_constexpr(tile_sizes);
  // one_tile_per_cta = tiles_per_cta==1
  bool one_tile_per_cta = (tiles_per_cta == 1);
  signature.append(std::to_string(one_tile_per_cta));
  stk.save_constexpr(one_tile_per_cta);

  void* global_scratch = nullptr;
  kernel_params.push_back(&global_scratch);

  // get function
  std::array<bool, 2> is_scalar;
  pointwise_dynamic::checkIfScalar(a_, b_, is_scalar);
  std::optional<TritonJITFunction> f;
  // TODO: code gen in c++

  auto ans_tuple = gen_add(ndim);
  std::string kernel_name = std::get<0>(ans_tuple);
  std::string file_path = std::get<1>(ans_tuple);

  // TODO: 四种情况
  if (!is_scalar[0] && !is_scalar[1]) {
    f = TritonJITFunction::getInstance(file_path, kernel_name);
  } else if (!is_scalar[0] && is_scalar[1]) {
    // TODO
    f = TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "add.py"),
                                       "add_func_tensor_scalar");
  } else if (is_scalar[0] && !is_scalar[1]) {
    // TODO
    f = TritonJITFunction::getInstance(std::string(utils::get_flag_gems_src_path() / "ops" / "add.py"),
                                       "add_func_scalar_tensor");
  } else {
    std::cout << "else";
    return a_ + b_;
  }
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  stk.save_tensor(a_);
  stk.save_tensor(b_);
  stk.save_tensor(out);
  // const expr需要在这里...
  stk.build();
  std::cout << "size of params" << kernel_params.size() << std::endl;

  std::cout << "file_path:" << file_path << std::endl;
  std::cout << "signature:" << signature << std::endl;

  std::cout << "--- Launching with raw args ---" << std::endl;
  std::cout << "raw_stream: " << raw_stream << std::endl;
  std::cout << "num_ctas: " << num_ctas << std::endl;
  std::cout << "num_warps: " << num_warps << std::endl;
  std::cout << "num_stages: " << num_stages << std::endl;
  std::cout << "signature: " << signature << std::endl;
  std::cout << "params: " << kernel_params << std::endl;
  f->launch_with_raw_args(raw_stream,
                          num_ctas,
                          1,
                          1,
                          num_warps,
                          num_stages,
                          // stk.get_signature(),
                          // stk.get_params()
                          signature,
                          kernel_params.data());
  return out;
}

};  // namespace flag_gems
