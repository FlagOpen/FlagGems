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
using ShapeR = c10::IntArrayRef;
using ShapeW = std::vector<long>;
using StrideR = c10::IntArrayRef;
using StrideW = std::vector<long>;
bool all_the_same_shape(const std::vector<at::Tensor>& tensors);
bool all_c_contiguous(const std::vector<at::Tensor>& tensors);
bool use_fast_path(const std::vector<at::Tensor>& tensors);
void checkIfScalar(const torch::Tensor& tensor1,
                   const torch::Tensor& tensor2,
                   std::array<bool, 2>& is_tensor);
ShapeW broadcast(const ShapeR& s1, const ShapeR& s2);
ShapeW broadcast_shapes(const std::vector<ShapeR>& shapes);
StrideW broadcasted_stride(const ShapeR& shape, const StrideR& stride, const ShapeR& new_shape);
void print_shapes(const std::vector<ShapeR>& shapes);
StrideW stride_order(const StrideR& strides);
StrideR create_stride_r_view(const StrideW& stride_w);
class StridedBuffer {
 public:
  StridedBuffer(const torch::Tensor& base,
                c10::optional<c10::IntArrayRef> shape = c10::nullopt,
                c10::optional<c10::IntArrayRef> strides = c10::nullopt,
                int64_t offset = 0);

  const c10::IntArrayRef strides() const;
  const c10::IntArrayRef sizes() const;
  long numel() const;
  int64_t dim() const;
  const torch::Tensor& unwrap() const;
  void* data_ptr() const;
  torch::Storage untyped_storage() const;
  StridedBuffer clone() const;
  StridedBuffer& copy_(const StridedBuffer& src);
  StridedBuffer& copy_(const torch::Tensor& src);
  long offset() const;

 private:
  torch::Tensor base_;
  void* data_ptr_;
  int64_t offset_;
  std::vector<long> shape_;
  std::vector<long> strides_;
  int64_t ndim_;
};
};  // namespace flag_gems::pointwise_dynamic
