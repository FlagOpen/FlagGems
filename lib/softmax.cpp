#include "c10/cuda/CUDAStream.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace {

// Get Triton JIT kernel
const TritonJITFunction& get_kernel(const std::string& name) {
  static const std::string src_path = (utils::get_triton_src_path() / "softmax.py").string();
  return TritonJITFunction::getInstance(src_path, name);
}

// General function to compute M, N, K dimensions
void compute_mnk(const at::Tensor& tensor, int dim, int64_t& M, int64_t& N, int64_t& K) {
  const auto sizes = tensor.sizes();
  M = 1, N = sizes[dim], K = 1;
  for (int i = 0; i < dim; ++i) M *= sizes[i];
  for (int i = dim + 1; i < sizes.size(); ++i) K *= sizes[i];
}

// Softmax forward kernel invocation
at::Tensor softmax_forward(const at::Tensor& input, int dim) {
  TORCH_CHECK(input.dim() >= 2, "Softmax input must be at least 2D");
  TORCH_CHECK(dim >= 0 && dim < input.dim(), "Softmax dim out of range");

  at::Tensor output = at::empty_like(input, input.options());

  int64_t M, N, K;
  compute_mnk(input, dim, M, N, K);

  constexpr int BLOCK_SIZE = 128;

  c10::DeviceGuard guard(input.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  int num_warps = 4;
  unsigned int num_ctas = 1;

  if (K == 1) {
    const TritonJITFunction& kernel = get_kernel("softmax_kernel_inner");

    kernel(
      raw_stream, M, 1, 1,
      num_warps, num_ctas,
      output, input,
      output.stride(dim - 1 >= 0 ? dim - 1 : 0), output.stride(dim),
      input.stride(dim - 1 >= 0 ? dim - 1 : 0), input.stride(dim),
      M, N,
      BLOCK_SIZE
    );
  } else {
    const TritonJITFunction& kernel = get_kernel("softmax_kernel_non_inner");

    kernel(
      raw_stream, M, K, 1,
      num_warps, num_ctas,
      output, input,
      output.stride(0), output.stride(dim), (dim + 1 < input.dim() ? output.stride(dim + 1) : 1),
      input.stride(0), input.stride(dim), (dim + 1 < input.dim() ? input.stride(dim + 1) : 1),
      M, N, K,
      BLOCK_SIZE
    );
  }

  return output;
}

void compute_mnk_for_backward(const at::Tensor& tensor,
                              int dim,
                              int64_t& M, int64_t& N, int64_t& K,
                              int64_t& stride_m, int64_t& stride_n, int64_t& stride_k) {
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();

  M = 1;
  for (int i = 0; i < dim; ++i) {
    M *= sizes[i];
  }

  N = sizes[dim];

  K = 1;
  for (int i = dim + 1; i < sizes.size(); ++i) {
    K *= sizes[i];
  }

  // Modified line: don't multiply by sizes[dim - 1]
  stride_m = (dim == 0) ? 0 : strides[dim - 1];  // Correct row stride

  stride_n = strides[dim];
  stride_k = (dim + 1 < sizes.size()) ? strides[dim + 1] : 1;

  if (K == 1) stride_k = 0;
  if (M == 1) stride_m = 0;
}

// Softmax backward kernel invocation
at::Tensor softmax_backward_impl(const at::Tensor& output,
                                 const at::Tensor& grad_output,
                                 int dim) {
  std::cout << "output.dim() = " << output.dim() << std::endl;

  int wrapped_dim = at::maybe_wrap_dim(dim, output.dim());
  std::cout << "softmax_backward_impl called with dim=" << dim
            << ", wrapped_dim=" << wrapped_dim
            << ", output.dim()=" << output.dim() << std::endl;

  TORCH_CHECK(wrapped_dim >= 0 && wrapped_dim < output.dim(), "wrapped_dim out of range!");

  at::Tensor grad_input = at::empty_like(grad_output, grad_output.options());

  int64_t M, N, K;
  int64_t stride_m, stride_n, stride_k;
  compute_mnk_for_backward(output, wrapped_dim, M, N, K, stride_m, stride_n, stride_k);

  std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;
  std::cout << "stride_m=" << stride_m << ", stride_n=" << stride_n << ", stride_k=" << stride_k << std::endl;

  constexpr int BLOCK_SIZE = 128;

  c10::DeviceGuard guard(output.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  int num_warps = 4;
  unsigned int num_ctas = 1;

  if (K == 1) {
    const TritonJITFunction& kernel = get_kernel("softmax_backward_kernel_inner");

    kernel(
      raw_stream, M, 1, 1,
      num_warps, num_ctas,
      output, grad_output, grad_input,
      stride_m, stride_n,
      M, N,
      BLOCK_SIZE
    );
  } else {
    const TritonJITFunction& kernel = get_kernel("softmax_backward_kernel_non_inner");

    kernel(
      raw_stream, M, K, 1,
      num_warps, num_ctas,
      output, grad_output, grad_input,
      stride_m, stride_n, stride_k,
      M, N, K,
      BLOCK_SIZE
    );
  }

  return grad_input;
}

}  // namespace

// Public API: softmax
at::Tensor softmax(const at::Tensor& input, int64_t dim, bool half_to_float) {
  at::Tensor input_tensor = input;
  if (half_to_float && input.scalar_type() == at::kHalf) {
    input_tensor = input_tensor.to(at::kFloat);
  }

  at::Tensor output = softmax_forward(input_tensor, static_cast<int>(dim));

  if (half_to_float && input.scalar_type() == at::kHalf) {
    output = output.to(at::kHalf);
  }

  return output;
}
// Public API: softmax backward
at::Tensor softmax_backward(const at::Tensor& grad_output,
                            const at::Tensor& output,
                            int64_t dim,
                            at::ScalarType input_dtype) {
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, output.dim());

  at::Tensor output_tensor = output;
  at::Tensor grad_output_tensor = grad_output;

  if (input_dtype == at::kHalf) {
    output_tensor = output.to(at::kFloat);
    grad_output_tensor = grad_output.to(at::kFloat);
  }

  at::Tensor grad_input = softmax_backward_impl(output_tensor, grad_output_tensor, wrapped_dim);

  if (grad_input.scalar_type() != input_dtype) {
    grad_input = grad_input.to(input_dtype);
  }

  return grad_input;
}



}  // namespace flag_gems
