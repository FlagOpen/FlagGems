#include "flag_gems/operators.h"
#include "torch/torch.h"
#include <gtest/gtest.h>

TEST(norm_op_test, rms_norm) {
  torch::manual_seed(0);
  const torch::Device device(torch::kCUDA, 0);

  torch::Tensor input = torch::randn({4, 8}, torch::TensorOptions().device(device).dtype(torch::kFloat16));
  torch::Tensor weight = torch::randn({8}, torch::TensorOptions().device(device).dtype(torch::kFloat16));
  double eps = 1e-5;
  //out_torch
  auto rms = input.pow(2).mean(-1, true).add(eps).sqrt();
  auto normed = input / rms;
  torch::Tensor out_torch =  normed * weight;

  //out triton
  torch::Tensor out_triton = flag_gems::rms_norm(input, weight, 1e-5);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton));
}
