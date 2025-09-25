#include <gtest/gtest.h>
#include <tuple>
#include "flag_gems/operators.h"
#include "torch/torch.h"

namespace {

torch::Tensor compute_reference(const torch::Tensor& x, const torch::Tensor& y) {
  auto result_dtype = torch::result_type(x, y);
  auto x_fp32 = x.to(torch::kFloat32);
  auto y_fp32 = y.to(torch::kFloat32);
  auto silu = torch::sigmoid(x_fp32) * x_fp32;
  auto out = silu * y_fp32;
  return out.to(result_dtype);
}

double atol_for(torch::Dtype dtype) {
  if (dtype == torch::kFloat32) {
    return 5e-5;
  }
  return 5e-3;
}

double rtol_for(torch::Dtype dtype) {
  if (dtype == torch::kFloat32) {
    return 5e-4;
  }
  return 5e-3;
}

}  // namespace

class SiluAndMulDTypeTest : public ::testing::TestWithParam<std::tuple<torch::Dtype, torch::Dtype>> {};

TEST_P(SiluAndMulDTypeTest, MatchesTorchReference) {
  torch::manual_seed(0);
  const torch::Device device(torch::kCUDA, 0);
  auto [x_dtype, y_dtype] = GetParam();

  auto x = torch::randn({4, 8}, torch::TensorOptions().device(device).dtype(x_dtype));
  auto y = torch::randn({4, 8}, torch::TensorOptions().device(device).dtype(y_dtype));

  auto ref = compute_reference(x, y);
  auto got = flag_gems::silu_and_mul(x, y);

  auto out_dtype = ref.scalar_type();
  EXPECT_EQ(got.scalar_type(), out_dtype);
  EXPECT_TRUE(torch::allclose(got, ref, rtol_for(out_dtype), atol_for(out_dtype)))
      << "Mismatch for dtypes (" << x_dtype << ", " << y_dtype << ")";
}

TEST_P(SiluAndMulDTypeTest, OutVariantWritesResult) {
  torch::manual_seed(1);
  const torch::Device device(torch::kCUDA, 0);
  auto [x_dtype, y_dtype] = GetParam();

  auto x = torch::randn({2, 5}, torch::TensorOptions().device(device).dtype(x_dtype));
  auto y = torch::randn({2, 5}, torch::TensorOptions().device(device).dtype(y_dtype));

  auto ref = compute_reference(x, y);
  auto out = torch::empty_like(ref);

  auto& ret = flag_gems::silu_and_mul_out(out, x, y);
  EXPECT_EQ(ret.data_ptr(), out.data_ptr());
  EXPECT_TRUE(torch::allclose(out, ref, rtol_for(ref.scalar_type()), atol_for(ref.scalar_type())));
}

TEST(SiluAndMulBroadcastTest, BroadcastsInputs) {
  torch::manual_seed(2);
  const torch::Device device(torch::kCUDA, 0);

  auto x = torch::randn({3, 4, 8}, torch::TensorOptions().device(device).dtype(torch::kFloat16));
  auto y = torch::randn({1, 4, 1}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

  auto ref = compute_reference(x, y);
  auto got = flag_gems::silu_and_mul(x, y);

  EXPECT_EQ(got.sizes(), ref.sizes());
  EXPECT_TRUE(torch::allclose(got, ref, rtol_for(ref.scalar_type()), atol_for(ref.scalar_type())));
}

INSTANTIATE_TEST_SUITE_P(DTypePairs,
                         SiluAndMulDTypeTest,
                         ::testing::Values(std::make_tuple(torch::kFloat16, torch::kFloat16),
                                           std::make_tuple(torch::kFloat16, torch::kFloat32),
                                           std::make_tuple(torch::kBFloat16, torch::kFloat16),
                                           std::make_tuple(torch::kFloat32, torch::kFloat32),
                                           std::make_tuple(torch::kBFloat16, torch::kBFloat16)));
