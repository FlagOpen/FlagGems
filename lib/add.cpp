#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

/*
at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_) {
  auto res = torch::broadcast_tensors({a_, b_});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor &a = res[0];
  const at::Tensor &b = res[1];

  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty(a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));

  const TritonJITFunction &f =
      TritonJITFunction::getInstance(std::string(utils::get_triton_src_path() / "binary_add.py"),
                                     "binary_pointwise_kernel");

  // add utility to build this automatically
  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;
  int64_t n = out.numel();
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(stream, num_blocks, 1, 1, num_warps, num_stages, a, b, out, n, tile_size);
  return out;
}
*/

at::Tensor add_tensor(const at::Tensor& a_, const at::Tensor& b_) {
  // 1. Broadcasting and ensuring contiguous memory layout
  auto res = torch::broadcast_tensors({a_, b_});
  const at::Tensor& a = res[0].contiguous();
  const at::Tensor& b = res[1].contiguous();

  // 2. Determine output dtype and create output tensor
  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty(a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));

  // 3. Get the TritonJITFunction instance
  const TritonJITFunction& f =
      TritonJITFunction::getInstance(std::string(utils::get_triton_src_path() / "binary_add.py"),
                                     "binary_pointwise_kernel");

  // 4. Manually prepare the raw argument list (void**) and the signature
  int64_t tile_size = 1024;
  int64_t n = out.numel();

  // This is the raw C-style argument array that the CUDA kernel expects.
  // It contains the addresses of all the kernel's parameters.
  std::vector<void*> raw_args_list;

  // Push the data pointers for the tensors.
  raw_args_list.push_back(a.data_ptr());
  raw_args_list.push_back(b.data_ptr());
  raw_args_list.push_back(out.data_ptr());

  // Push the addresses of scalar values.
  // NOTE: The scalars 'n' and 'tile_size' must have their addresses taken.
  // This is why we use references or variables.
  raw_args_list.push_back(&n);
  raw_args_list.push_back(&tile_size);

  // 5. Manually generate the signature string
  // This must match the kernel's type-based signature for overload resolution.
  // This is an example; the exact signature depends on the kernel definition.
  std::string signature = "tl.pointer_type,tl.pointer_type,tl.pointer_type,int64,int64";

  // 6. Set up the launch configuration
  const int num_warps = 8;
  const int num_stages = 1;
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  // 7. Launch the kernel using the raw argument list
  f.launch_with_raw_args(raw_stream,
                         num_blocks,
                         1,
                         1,
                         num_warps,
                         num_stages,
                         signature,
                         raw_args_list.data());

  return out;
}

}  // namespace flag_gems
