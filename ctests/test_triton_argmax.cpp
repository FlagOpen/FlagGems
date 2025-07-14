#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(reduction_op_test, argmax) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::randn({4, 4}, device);

  torch::Tensor ref_output = at::argmax(input);
  torch::Tensor triton_output = flag_gems::argmax(input);

  EXPECT_TRUE(torch::equal(ref_output, triton_output));
}

TEST(reduction_op_test, argmax_dim_specific) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::randn({3, 5, 7}, device);

  torch::Tensor ref_dim0 = at::argmax(input, 0);
  torch::Tensor triton_dim0 = flag_gems::argmax(input, 0);

  EXPECT_TRUE(torch::equal(ref_dim0, triton_dim0));

  torch::Tensor ref_dim1 = at::argmax(input, -1);
  torch::Tensor triton_dim1 = flag_gems::argmax(input, -1);

  EXPECT_TRUE(torch::equal(ref_dim1, triton_dim1));
}

TEST(reduction_op_test, argmax_keepdim_option) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor input = torch::randn({2, 2, 2, 2}, device);

  torch::Tensor ref_keep = at::argmax(input, 1, true);
  torch::Tensor triton_keep = flag_gems::argmax(input, 1, true);

  EXPECT_TRUE(torch::equal(ref_keep, triton_keep));
  EXPECT_EQ(ref_keep.sizes(), triton_keep.sizes());

  torch::Tensor ref_no_keep = at::argmax(input, 1, false);
  torch::Tensor triton_no_keep = flag_gems::argmax(input, 1, false);

  EXPECT_TRUE(torch::equal(ref_no_keep, triton_no_keep));
}
