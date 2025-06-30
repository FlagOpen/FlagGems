#include <gtest/gtest.h>

#include "flag_gems/operators.h"
#include "torch/torch.h"
TEST(zeros_op_test, zeros) {
  const torch::Device device(torch::kCUDA, 0);
  int64_t n = 100;
  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  torch::Tensor out_torch = torch::zeros({n}, options);
  torch::Tensor out_triton_0 = flag_gems::zeros_tensor(n);
  torch::Tensor out_triton_1 = flag_gems::zeros_tensor(n, torch::kFloat32);
  torch::Tensor out_triton_2 = flag_gems::zeros_tensor(n, torch::kFloat32, c10::nullopt);

  EXPECT_TRUE(torch::allclose(out_torch, out_triton_0));
  EXPECT_TRUE(torch::allclose(out_torch, out_triton_1));
  EXPECT_TRUE(torch::allclose(out_torch, out_triton_2));
}
