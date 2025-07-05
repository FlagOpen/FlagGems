#include "flag_gems/operators.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

TEST(FillTest, ScalarFill) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({4, 5}, torch::TensorOptions().device(device));
  double val = 3.14;

  torch::Tensor out_gems = flag_gems::fill_scalar(t, val);
  torch::Tensor out_ref = torch::full_like(t, val);

  EXPECT_TRUE(torch::allclose(out_gems, out_ref));
}

TEST(FillTest, TensorFill) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({3, 3}, torch::TensorOptions().device(device));
  torch::Tensor val = torch::tensor(7.5, torch::TensorOptions().device(device));

  torch::Tensor out_gems = flag_gems::fill_tensor(t, val);
  torch::Tensor out_ref = torch::full_like(t, val.item<double>());

  EXPECT_TRUE(torch::allclose(out_gems, out_ref));
}

TEST(FillTest, ScalarFillInplace) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({2, 2}, torch::TensorOptions().device(device));
  double val = -1.23;

  flag_gems::fill_scalar_(t, val);
  torch::Tensor ref = torch::full_like(t, val);

  EXPECT_TRUE(torch::allclose(t, ref));
}

TEST(FillTest, TensorFillInplace) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({2, 2}, torch::TensorOptions().device(device));
  torch::Tensor val = torch::tensor(-2.5, torch::TensorOptions().device(device));

  flag_gems::fill_tensor_(t, val);
  torch::Tensor ref = torch::full_like(t, val.item<double>());

  EXPECT_TRUE(torch::allclose(t, ref));
}

TEST(FillTest, EmptyTensor) {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor t = torch::empty({0}, torch::TensorOptions().device(device));
  double val = 42.0;

  torch::Tensor out_gems = flag_gems::fill_scalar(t, val);
  torch::Tensor out_ref = torch::full_like(t, val);

  EXPECT_EQ(out_gems.numel(), 0);
  EXPECT_TRUE(torch::allclose(out_gems, out_ref));
}

TEST(FillTest, DifferentDtypes) {
  const torch::Device device(torch::kCUDA, 0);

  auto check_dtype = [&](auto dtype) {
    torch::Tensor t = torch::empty({5, 5}, torch::TensorOptions().device(device).dtype(dtype));
    double val = 123.0;
    torch::Tensor out = flag_gems::fill_scalar(t, val);
    torch::Tensor ref = torch::full_like(t, val);
    EXPECT_TRUE(torch::allclose(out.to(torch::kFloat32), ref.to(torch::kFloat32)));
  };

  check_dtype(torch::kFloat32);
  check_dtype(torch::kFloat64);
  check_dtype(torch::kInt32);
  check_dtype(torch::kInt64);
}
