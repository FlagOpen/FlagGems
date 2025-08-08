#pragma once
#include <dlfcn.h>  // dladdr
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "torch/torch.h"

namespace flag_gems::utils {
using Shape = std::vector<long>;
std::filesystem::path get_path_of_this_library();
std::filesystem::path get_triton_src_path();
std::filesystem::path get_flag_gems_src_path();
int64_t next_power_of_2(int64_t n);
std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(const at::Tensor& tensor,
                                                                      int reduction_axis);
std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(
    const at::Tensor& tensor, at::OptionalIntArrayRef reduction_axes_opt);
std::tuple<int64_t, int64_t, int64_t> parse_reduction_axes(const at::Tensor& tensor, int reduction_axis);
int cdiv(int a, int b);
bool broadcastable_to(at::IntArrayRef s1, at::IntArrayRef s2);
};  // namespace flag_gems::utils

namespace flag_gems::pointwise_dynamic {
using Shape = std::vector<long>;
using Stride = std::vector<long>;
Stride broadcasted_stride(const Shape& shape, const Stride& stride, const Shape& new_shape);
bool all_the_same_shape(const std::vector<at::Tensor>& tensors);
bool all_c_contiguous(const std::vector<at::Tensor>& tensors);
bool use_fast_path(const std::vector<at::Tensor>& tensors);
void checkIfScalar(const torch::Tensor& tensor1,
                   const torch::Tensor& tensor2,
                   std::array<bool, 2>& is_tensor);
};  // namespace flag_gems::pointwise_dynamic
