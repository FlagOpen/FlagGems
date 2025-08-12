#include "flag_gems/utils.h"

namespace flag_gems::utils {

std::filesystem::path get_path_of_this_library() {
  // This function gives the library path of this library as runtime, similar to the $ORIGIN
  // that is used for run path (RPATH), but unfortunately, for custom dependencies (instead of linking)
  // there is no build system generator to take care of this.
  static const std::filesystem::path cached_path = []() {
    Dl_info dl_info;
    if (dladdr(reinterpret_cast<void*>(&get_path_of_this_library), &dl_info) && dl_info.dli_fname) {
      return std::filesystem::canonical(dl_info.dli_fname);  // Ensure absolute, resolved path
    } else {
      throw std::runtime_error("cannot get the path of libjit_utils.so");
    }
  }();
  return cached_path;
}

std::filesystem::path get_triton_src_path() {
  const static std::filesystem::path triton_src_dir = []() {
    std::filesystem::path installed_script_path =
        get_path_of_this_library().parent_path().parent_path() / "share" / "flag_gems" / "triton_src";
    if (std::filesystem::exists(installed_script_path)) {
      return installed_script_path;
    } else {
      std::filesystem::path source_script_path =
          std::filesystem::path(__FILE__).parent_path().parent_path() / "triton_src";
      return source_script_path;
    }
  }();
  return triton_src_dir;
}

std::filesystem::path get_flag_gems_src_path() {
  const static std::filesystem::path flag_gems_src_dir = []() {
    const char* flag_gems_dir = std::getenv("FLAGGEMS_SOURCE_DIR");
    if (!flag_gems_dir) {
      throw std::runtime_error("Environment variable FLAGGEMS_SOURCE_DIR not set");
    }
    return std::filesystem::path(flag_gems_dir);
  }();
  return flag_gems_src_dir;
}

std::filesystem::path get_code_cache_dir() {
  const char* env_cache_dir = std::getenv("FLAGGEMS_CACHE_DIR");
  std::filesystem::path cache_dir;
  if (env_cache_dir) {
    cache_dir = std::filesystem::path(env_cache_dir);
  } else {
    cache_dir = std::filesystem::path(std::getenv("HOME")) / ".flaggems";
  }
  std::filesystem::create_directories(cache_dir);
  std::filesystem::path code_cache_dir = cache_dir / "code_cache";
  std::filesystem::create_directories(code_cache_dir);
  return code_cache_dir;
}

int64_t next_power_of_2(int64_t n) {
  if (n <= 1) return 1;
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}

bool broadcastable_to(at::IntArrayRef s1, at::IntArrayRef s2) {
  size_t r1 = s1.size();
  if (r1 == 0) {
    return true;
  }

  size_t r2 = s2.size();
  if (r2 == 0) {
    return false;
  }

  if (r1 > r2) {
    return false;
  }

  size_t d = r2 - r1;
  for (size_t i = 0; i < r1; ++i) {
    if (s1[i] == 1 || s1[i] == s2[d + i]) {
      continue;
    }
    return false;
  }

  return true;
}

std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(
    const at::Tensor& tensor, at::OptionalIntArrayRef reduction_axes_opt) {
  int64_t dim = tensor.dim();
  c10::DimVector reduction_axes;

  if (reduction_axes_opt.has_value()) {
    reduction_axes = reduction_axes_opt.value().vec();
  }

  std::unordered_set<int64_t> reduction_set(reduction_axes.begin(), reduction_axes.end());

  c10::DimVector left_axes, right_axes;
  int64_t non_reduction_size = 1, reduction_size = 1;

  for (int64_t i = 0; i < dim; ++i) {
    if (reduction_set.count(i)) {
      right_axes.push_back(i);
      reduction_size *= tensor.size(i);
    } else {
      left_axes.push_back(i);
      non_reduction_size *= tensor.size(i);
    }
  }

  // Concatenate left and right axes to form the new permutation order
  c10::DimVector permute_order = left_axes;
  permute_order.insert(permute_order.end(), right_axes.begin(), right_axes.end());

  return {tensor.permute(permute_order), non_reduction_size, reduction_size};
}
std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(const at::Tensor& tensor,
                                                                      int reduction_axis) {
  int64_t dim = tensor.dim();
  c10::DimVector left_axes, right_axes;
  int64_t non_reduction_size = 1, reduction_size = 1;

  for (int64_t i = 0; i < dim; ++i) {
    if (i == reduction_axis) {
      right_axes.push_back(i);
      reduction_size *= tensor.size(i);
    } else {
      left_axes.push_back(i);
      non_reduction_size *= tensor.size(i);
    }
  }
  c10::DimVector permute_order = left_axes;
  permute_order.insert(permute_order.end(), right_axes.begin(), right_axes.end());

  return {tensor.permute(permute_order), non_reduction_size, reduction_size};
}

std::tuple<int64_t, int64_t, int64_t> parse_reduction_axes(const at::Tensor& tensor, int reduction_axis) {
  int64_t dim = tensor.dim();
  c10::DimVector left_axes, right_axes, remain_axes;
  int64_t non_reduction_size = 1;
  int64_t reduction_size = 1;
  int64_t remain_size = 1;

  for (int64_t i = 0; i < dim; ++i) {
    if (i < reduction_axis) {
      left_axes.push_back(i);
      non_reduction_size *= tensor.size(i);
    } else if (i == reduction_axis) {
      right_axes.push_back(i);
      reduction_size *= tensor.size(i);
    } else {
      remain_axes.push_back(i);
      remain_size *= tensor.size(i);
    }
  }
  return {non_reduction_size, reduction_size, remain_size};
}
int cdiv(int a, int b) {
  return (a + b - 1) / b;
}
}  // namespace flag_gems::utils

namespace flag_gems::pointwise_dynamic {
void checkIfScalar(const torch::Tensor& tensor1,
                   const torch::Tensor& tensor2,
                   std::array<bool, 2>& is_tensor) {
  is_tensor[0] = (tensor1.dim() == 0);
  is_tensor[1] = (tensor2.dim() == 0);
}
StridedBuffer::StridedBuffer(const torch::Tensor& base,
                             c10::optional<c10::IntArrayRef> shape,
                             c10::optional<c10::IntArrayRef> strides,
                             int64_t offset)
    : base_(base.contiguous()), offset_(offset) {
  if (offset_ == 0) {
    data_ptr_ = base_.data_ptr();
  } else {
    data_ptr_ = static_cast<char*>(base_.data_ptr()) + base_.element_size() * offset_;
  }
  shape_ = shape.has_value() ? shape.value().vec() : base_.sizes().vec();
  strides_ = strides.has_value() ? strides.value().vec() : base_.strides().vec();
  ndim_ = shape_.size();
}

const c10::IntArrayRef StridedBuffer::strides() const {
  return strides_;
}

const c10::IntArrayRef StridedBuffer::sizes() const {
  return shape_;
}

long StridedBuffer::numel() const {
  long num = 1;
  for (long s : shape_) {
    num *= s;
  }
  return num;
}

int64_t StridedBuffer::dim() const {
  return ndim_;
}

const torch::Tensor& StridedBuffer::unwrap() const {
  return base_;
}

void* StridedBuffer::data_ptr() const {
  return data_ptr_;
}

torch::Storage StridedBuffer::untyped_storage() const {
  return base_.storage();
}

StridedBuffer StridedBuffer::clone() const {
  torch::Tensor cloned_base = base_.clone();
  return StridedBuffer(cloned_base, shape_, strides_, offset_);
}

StridedBuffer& StridedBuffer::copy_(const StridedBuffer& src) {
  torch::Tensor temp_dst = torch::empty_like(src.unwrap());
  temp_dst.copy_(src.unwrap());

  base_ = temp_dst;
  strides_ = src.strides().vec();
  shape_ = src.sizes().vec();
  offset_ = src.offset();
  data_ptr_ = base_.data_ptr();

  return *this;
}

StridedBuffer& StridedBuffer::copy_(const torch::Tensor& src) {
  StridedBuffer src_buffer(src);
  return this->copy_(src_buffer);
}

long StridedBuffer::offset() const {
  return offset_;
}

ShapeW broadcast(const ShapeR& s1, const ShapeR& s2) {
  long ndim = std::max(s1.size(), s2.size());
  ShapeW output_shape(ndim);
  long p1 = s1.size() - 1;
  long p2 = s2.size() - 1;

  for (long i = ndim - 1; i >= 0; --i) {
    long d1 = (p1 >= 0) ? s1[p1] : 1;
    long d2 = (p2 >= 0) ? s2[p2] : 1;

    if (d1 != d2 && d1 != 1 && d2 != 1) {
      // 抛出异常或返回错误，因为形状不可广播
      throw std::runtime_error("Shapes are not broadcastable.");
    }
    output_shape[i] = std::max(d1, d2);
    if (p1 >= 0) p1--;
    if (p2 >= 0) p2--;
  }
  return output_shape;
}

ShapeW broadcast_shapes(const std::vector<ShapeR>& shapes) {
  if (shapes.empty()) {
    return {};
  }

  ShapeW output_shape(shapes[0].begin(), shapes[0].end());
  for (size_t i = 1; i < shapes.size(); ++i) {
    output_shape = broadcast(output_shape, shapes[i]);
  }
  return output_shape;
}

StrideW broadcasted_stride(const ShapeR& shape, const StrideR& stride, const ShapeR& new_shape) {
  assert(broadcastable_to(shape, new_shape) && "Shapes are not broadcastable.");

  int r1 = shape.size();
  int r2 = new_shape.size();
  int d = r2 - r1;

  StrideW new_stride(r2, 0);
  for (int i = 0; i < r1; ++i) {
    int new_dim_index = d + i;
    if (shape[i] == 1 && new_shape[new_dim_index] > 1) {
      new_stride[new_dim_index] = 0;
    } else {
      new_stride[new_dim_index] = stride[i];
    }
  }
  return new_stride;
}

bool all_the_same_shape(const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    return true;
  }
  const auto& first_shape = tensors[0].sizes();
  for (const auto& tensor : tensors) {
    if (!tensor.sizes().equals(first_shape)) {
      return false;
    }
  }
  return true;
}

bool all_c_contiguous(const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    return true;
  }
  for (const auto& tensor : tensors) {
    if (!tensor.is_contiguous()) {
      return false;
    }
  }
  return true;
}

bool all_the_same_stride(const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    return true;
  }
  const auto& first_stride = tensors[0].strides();
  for (const auto& tensor : tensors) {
    if (!tensor.strides().equals(first_stride)) {
      return false;
    }
  }
  return true;
}
bool use_fast_path(const std::vector<at::Tensor>& tensors) {
  if (!all_the_same_shape(tensors)) {
    return false;
  }
  if (all_c_contiguous(tensors)) {
    return true;
  }
  return all_the_same_stride(tensors) && tensors[0].is_non_overlapping_and_dense();
}
StrideW stride_order(const StrideR& strides) {
  // Create a vector of indices from 0 to strides.size() - 1
  StrideW indices(strides.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Sort the indices based on the absolute value of the corresponding stride
  std::sort(indices.begin(), indices.end(), [&](int64_t i, int64_t j) {
    return std::abs(strides[i]) < std::abs(strides[j]);
  });

  return indices;
}

StrideR create_stride_r_view(const StrideW& stride_w) {
  return StrideR(reinterpret_cast<const int64_t*>(stride_w.data()), stride_w.size());
}
};  // namespace flag_gems::pointwise_dynamic
