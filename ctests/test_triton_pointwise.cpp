#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(pointwise_op_simple_test, add) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({128}, device);
  torch::Tensor b = torch::randn({128}, device);

  torch::Tensor out_torch = a + b;
  torch::Tensor out_triton = flag_gems::add_tensor(a, b);
  std::cout << "out_torch sizes: " << out_torch.sizes() << std::endl;
  std::cout << "out_triton sizes: " << out_triton.sizes() << std::endl;
  EXPECT_TRUE(torch::allclose(out_torch, out_triton));
}
