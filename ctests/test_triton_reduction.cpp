#include <gtest/gtest.h>
#include "c10/util/Logging.h"
#include "flag_gems/operators.h"
#include "torch/torch.h"
TEST(reduction_op_test, sum) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({32, 1024}, device);

  torch::Tensor out_torch = at::sum(a, {1});
  torch::Tensor out_triton = flag_gems::sum_dim(a, {1});

  if (!torch::allclose(out_torch, out_triton, 1e-5, 1e-8)) {
    LOG(INFO) << "Difference:\n" << out_torch - out_triton;
  }

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-5, 1e-8));
}

TEST(reduction_op_test, nonzero) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({32, 1024}, device);
  a = a > 0.5;

  torch::Tensor out_torch = at::nonzero(a);
  torch::Tensor out_triton = flag_gems::nonzero(a);

  if (!torch::allclose(out_torch, out_triton, 1e-5, 1e-8)) {
    LOG(INFO) << "Difference:\n" << out_torch - out_triton;
  }

  EXPECT_TRUE(torch::allclose(out_torch, out_triton, 1e-5, 1e-8));
}

TEST(reduction_op_test, max_keepdim) {
    const torch::Device device(torch::kCUDA, 0);
    for (bool keepdim : {false, true}) {
        torch::Tensor a = torch::randn({32, 1024}, device);
        auto out_torch = at::max(a, 1, keepdim); 
        torch::Tensor max_torch = std::get<0>(out_torch);
        torch::Tensor index_torch = std::get<1>(out_torch);
        auto out_triton = flag_gems::max_dim(a, 1, keepdim, std::nullopt); 
        torch::Tensor max_triton = std::get<0>(out_triton);
        torch::Tensor index_triton = std::get<1>(out_triton);
        LOG(INFO) << "Input shape: " << a.sizes();
        LOG(INFO) << "Max Torch shape: " << max_torch.sizes();
        LOG(INFO) << "Index Torch shape: " << index_torch.sizes();
        LOG(INFO) << "Max Triton shape (keepdim=" << keepdim << "): " << max_triton.sizes();
        LOG(INFO) << "Index Triton shape (keepdim=" << keepdim << "): " << index_triton.sizes();
        if (!torch::allclose(max_torch, max_triton, 1e-5, 1e-8)) {
            LOG(INFO) << "Max value difference (keepdim=" << keepdim << "):\n" << max_torch - max_triton;
        }
        EXPECT_TRUE(torch::allclose(max_torch, max_triton, 1e-5, 1e-8));
        if (!torch::allclose(index_torch, index_triton, 1e-5, 1e-8)) {
            LOG(INFO) << "Index difference (keepdim=" << keepdim << "):\n" << index_torch - index_triton;
        }
        EXPECT_TRUE(torch::allclose(index_torch, index_triton, 1e-5, 1e-8));
    }
}