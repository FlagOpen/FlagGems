#include <pybind11/pybind11.h>
#include "flag_gems/operators.h"

PYBIND11_MODULE(c_operators, m) {
  // 这里暂时空，PyTorch 扩展用 TORCH_LIBRARY 方式注册算子
}

namespace flag_gems {

// 先声明算子接口
TORCH_LIBRARY(flag_gems, m) {
  m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor", {at::Tag::pt2_compliant_tag});
  // Norm
  m.def("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()");
  // Rotary embedding
  m.def(
      "rotary_embedding_inplace(Tensor! q, Tensor! k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> ()");
  m.def(
      "rotary_embedding(Tensor q, Tensor k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> (Tensor, Tensor)");

  // ** 新增 fill 算子声明 **
  m.def("fill_scalar(Tensor input, double value) -> Tensor");
  m.def("fill_tensor(Tensor input, Tensor value) -> Tensor");
  m.def("fill_scalar_(Tensor! input, double value) -> Tensor!");
  m.def("fill_tensor_(Tensor! input, Tensor value) -> Tensor!");
}

} // namespace flag_gems

// 注意：TORCH_LIBRARY_IMPL 不能写在命名空间内部或者函数内部，必须放全局作用域
TORCH_LIBRARY_IMPL(flag_gems, CUDA, m) {
  m.impl("sum.dim_IntList", TORCH_FN(flag_gems::sum_dim));
  m.impl("add_tensor", TORCH_FN(flag_gems::add_tensor));
  // Norm
  m.impl("rms_norm", TORCH_FN(flag_gems::rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(flag_gems::fused_add_rms_norm));
  // Rotary embedding
  m.impl("rotary_embedding", TORCH_FN(flag_gems::rotary_embedding));
  m.impl("rotary_embedding_inplace", TORCH_FN(flag_gems::rotary_embedding_inplace));

  // ** 新增 fill 算子实现绑定 **
  m.impl("fill_scalar", TORCH_FN(flag_gems::fill_scalar));
  m.impl("fill_tensor", TORCH_FN(flag_gems::fill_tensor));
  m.impl("fill_scalar_", TORCH_FN(flag_gems::fill_scalar_));
  m.impl("fill_tensor_", TORCH_FN(flag_gems::fill_tensor_));
}
