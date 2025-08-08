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

std::filesystem::path get_path_of_this_library();
std::filesystem::path get_triton_src_path();
std::filesystem::path get_flag_gems_src_path();
int64_t next_power_of_2(int64_t n);
bool broadcastable_to(at::IntArrayRef s1, at::IntArrayRef s2);
};  // namespace flag_gems::utils

namespace flag_gems::pointwise_dynamic {
using Shape = std::vector<long>;
using Stride = std::vector<long>;
bool broadcastable_to(const Shape& shape, const Shape& new_shape);
Stride broadcasted_stride(const Shape& shape, const Stride& stride, const Shape& new_shape);
bool all_the_same_shape(const std::vector<at::Tensor>& tensors);
bool all_c_contiguous(const std::vector<at::Tensor>& tensors);
bool use_fast_path(const std::vector<at::Tensor>& tensors);
};  // namespace flag_gems::pointwise_dynamic
