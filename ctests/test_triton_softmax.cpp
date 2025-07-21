#include "flag_gems/operators.h"
#include "gtest/gtest.h"
#include "torch/torch.h"
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

TEST(TritonSoftmaxTest, ForwardInnerDim) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({4, 16}, device).to(torch::kFloat16);

  // Call flag_gems softmax, assuming dim=1 is the inner dimension softmax
  auto out_gems = flag_gems::softmax(input, 1, false);
  auto out_torch = torch::softmax(input.to(torch::kFloat32), 1).to(torch::kFloat16);

  EXPECT_TRUE(torch::allclose(out_gems, out_torch, 1e-2, 1e-3));

  // Numerical check: each row sum should be close to 1
  auto row_sums = out_gems.sum(1);
  auto ones = torch::ones_like(row_sums);
  EXPECT_TRUE(torch::allclose(row_sums, ones, 1e-2, 1e-3));
}

TEST(TritonSoftmaxTest, ForwardNonInnerDim) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({2, 8, 3}, device).to(torch::kFloat16);

  // dim=1 is not the inner-most dimension
  auto out_gems = flag_gems::softmax(input, 1, false);
  auto out_torch = torch::softmax(input.to(torch::kFloat32), 1).to(torch::kFloat16);

  EXPECT_TRUE(torch::allclose(out_gems, out_torch, 1e-2, 1e-3));

  // Numerical check
  auto sums = out_gems.sum(1);
  auto ones = torch::ones_like(sums);
  EXPECT_TRUE(torch::allclose(sums, ones, 1e-2, 1e-3));
}

TEST(TritonSoftmaxTest, ForwardDim0) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({5, 10}, device).to(torch::kFloat16);

  // Test softmax along dim=0
  auto out_gems = flag_gems::softmax(input, 0, false);
  auto out_torch = torch::softmax(input.to(torch::kFloat32), 0).to(torch::kFloat16);

  EXPECT_TRUE(torch::allclose(out_gems, out_torch, 1e-2, 1e-3));

  // Numerical check: column-wise sum should be close to 1
  auto col_sums = out_gems.sum(0);
  auto ones = torch::ones_like(col_sums);
  EXPECT_TRUE(torch::allclose(col_sums, ones, 1e-2, 1e-3));
}

TEST(TritonSoftmaxTest, BackwardInnerDim) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({4, 16}, device).to(torch::kFloat32).set_requires_grad(true);
  int dim = 1;
  int wrapped_dim = at::maybe_wrap_dim(dim, input.dim());

  auto output_ref = torch::softmax(input, wrapped_dim);
  auto output_triton = flag_gems::softmax(input, wrapped_dim, false);

  auto grad_output = torch::randn_like(output_ref);

  torch::Tensor grad_input_ref;
  {
    pybind11::gil_scoped_release no_gil;  // Release GIL
    grad_input_ref = torch::autograd::grad({output_ref}, {input}, {grad_output})[0];
  }

  auto grad_input_triton = flag_gems::softmax_backward(grad_output, output_triton, wrapped_dim, input.scalar_type());

  EXPECT_TRUE(torch::allclose(grad_input_triton, grad_input_ref, 1e-2, 1e-2));
}

TEST(TritonSoftmaxTest, BackwardNonInnerDim) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({2, 8, 3}, device).to(torch::kFloat32).set_requires_grad(true);
  int dim = 1;
  int wrapped_dim = at::maybe_wrap_dim(dim, input.dim());

  auto output_ref = torch::softmax(input, wrapped_dim);
  auto output_triton = flag_gems::softmax(input, wrapped_dim, false);

  auto grad_output = torch::randn_like(output_ref);

  torch::Tensor grad_input_ref;
  {
    pybind11::gil_scoped_release no_gil;  // Release GIL
    grad_input_ref = torch::autograd::grad({output_ref}, {input}, {grad_output})[0];
  }

  auto grad_input_triton = flag_gems::softmax_backward(grad_output, output_triton, wrapped_dim, input.scalar_type());

  EXPECT_TRUE(torch::allclose(grad_input_triton, grad_input_ref, 1e-2, 1e-2));
}

TEST(TritonSoftmaxTest, BackwardDim0) {
  const torch::Device device(torch::kCUDA, 0);
  auto input = torch::randn({5, 10}, device).to(torch::kFloat32).set_requires_grad(true);
  int dim = 0;
  int wrapped_dim = at::maybe_wrap_dim(dim, input.dim());

  auto output_ref = torch::softmax(input, wrapped_dim);
  auto output_triton = flag_gems::softmax(input, wrapped_dim, false);

  auto grad_output = torch::randn_like(output_ref);

  torch::Tensor grad_input_ref;
  {
    pybind11::gil_scoped_release no_gil;  // Release GIL
    grad_input_ref = torch::autograd::grad({output_ref}, {input}, {grad_output})[0];
  }

  auto grad_input_triton = flag_gems::softmax_backward(grad_output, output_triton, dim, input.scalar_type());

  EXPECT_TRUE(torch::allclose(grad_input_triton, grad_input_ref, 1e-2, 1e-2));
}
