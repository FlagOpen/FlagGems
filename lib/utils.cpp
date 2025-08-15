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
                   std::array<bool, 2>& is_scalar) {
  is_scalar[0] = (tensor1.dim() == 0);
  is_scalar[1] = (tensor2.dim() == 0);
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

void ParamStack::save_tensor(const at::Tensor& tensor) {
  void* p_item = tensor.data_ptr();
  tensor_ptr.push_back(p_item);
  if (tensor.dtype() == at::kFloat) {
    kernel_params.push_back(&(tensor_ptr.back()));
    signature.append("*fp32:16,");
  } else if (tensor.dtype() == at::kInt) {
    kernel_params.push_back(&(tensor_ptr.back()));
    signature.append("*int32:16,");
  } else if (tensor.dtype() == at::kDouble) {
    kernel_params.push_back(&(tensor_ptr.back()));
    signature.append("*fp64:16,");
  } else if (tensor.dtype() == at::kHalf) {
    kernel_params.push_back(&(tensor_ptr.back()));
    signature.append("*fp16:16,");
  } else {
    std::runtime_error("TypeError: we only support fp64/32/16 and int32 now");
  }
}

void ParamStack::save_tensor(at::Tensor& tensor) {
  void* p_item = tensor.data_ptr();
  tensor_ptr.push_back(p_item);
  if (tensor.dtype() == at::kFloat) {
    kernel_params.push_back(&(tensor_ptr.back()));
    signature.append("*fp32:16,");
  } else if (tensor.dtype() == at::kInt) {
    kernel_params.push_back(&(tensor_ptr.back()));
    signature.append("*int32:16,");
  } else if (tensor.dtype() == at::kDouble) {
    kernel_params.push_back(&(tensor_ptr.back()));
    signature.append("*fp64:16,");
  } else if (tensor.dtype() == at::kHalf) {
    kernel_params.push_back(&(tensor_ptr.back()));
    signature.append("*fp16:16,");
  } else {
    std::runtime_error("TypeError: we only support fp64/32/16 and int32 now");
  }
}

std::string ParamStack::get_signature() {
  if (!signature.empty() && signature.back() == ',') {
    signature.pop_back();
  }
  return signature;
}

void** ParamStack::get_params() {
  void** res = kernel_params.empty() ? nullptr : kernel_params.data();
  if (res == nullptr) {
    // kernel_params 是空的
    std::cout << "The parameter stack is empty." << std::endl;
  } else {
    // kernel_params 不为空
    std::cout << "The parameter stack is not empty." << std::endl;
  }
  return res;
}

void ParamStack::save_stride(int64_t stride) {
  if (stride == 1) {
    strides.push_back(0);
  } else {
    strides.push_back(stride);
  }
}

void ParamStack::save_task_shape(int64_t shape) {
  task_shape.push_back(shape);
}

void ParamStack::save_task_partition(int64_t partition) {
  if (partition == 1) {
    task_partition.push_back(0);
  } else {
    task_partition.push_back(partition);
  }
}

void ParamStack::push_strides() {
  for (auto& stride : strides) {
    if (stride != 0) {
      kernel_params.push_back(static_cast<void*>(&stride));
      signature.append("i64,");
    } else {
      signature.append("i64:1,");
    }
  }
}

void ParamStack::push_task_shape() {
  for (auto& shape : task_shape) {
    kernel_params.push_back(static_cast<void*>(&shape));
    signature.append("i64,");
  }
}

void ParamStack::push_task_partition() {
  for (auto& partition : task_partition) {
    if (partition != 0) {
      kernel_params.push_back(static_cast<void*>(&partition));
      signature.append("i64,");
    } else {
      signature.append("i64:1,");
    }
  }
}

void ParamStack::add_global_scratch() {
  kernel_params.push_back(&global_scratch);
}

void ParamStack::build() {
  push_strides();
  push_task_shape();
  push_task_partition();
  signature.append(constexp);
  add_global_scratch();
}

void ParamStack::save_constexpr(int64_t value) {
  constexp.append(std::to_string(value) + ",");
}

void ParamStack::save_constexpr(bool value) {
  if (value) {
    constexp.append("True,");
  } else {
    constexp.append("False,");
  }
}

};  // namespace flag_gems::pointwise_dynamic
