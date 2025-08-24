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
using Shape = c10::IntArrayRef;
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
void checkIfScalar(const torch::Tensor& tensor1,
                   const torch::Tensor& tensor2,
                   std::array<bool, 2>& is_tensor);
bool use_fast_path(const std::vector<at::Tensor>& tensors);

class ParamStack {
 private:
  std::vector<void*> kernel_params;
  std::string signature;
  std::vector<void*> tensor_ptr;
  std::vector<int64_t> strides;
  std::vector<int64_t> task_shape;
  std::vector<int64_t> task_partition;
  std::string constexp;
  void* global_scratch;

 private:
  void push_strides();
  void push_task_shape();
  void push_task_partition();
  void add_global_scratch();

 public:
  ParamStack(int max_args = 32) {
    kernel_params.reserve(max_args);
    tensor_ptr.reserve(max_args);
    void* global_scratch = nullptr;
  }
  void save_tensor(at::Tensor& tensor);
  void save_tensor(const at::Tensor& tensor);
  void save_stride(int64_t stride);
  void save_task_shape(int64_t shape);
  void save_task_partition(int64_t partition);
  void save_constexpr(int64_t value);
  void save_constexpr(bool value);
  void** get_params();
  std::string get_signature();

  void build();
};
};  // namespace flag_gems::pointwise_dynamic
