#include <pybind11/pybind11.h>
#include "torch/python.h"

#include "flag_gems/operators.h"

// TODO: use pytorch's argparse utilities to generate CPython bindings, since it is more efficient than
// bindings provided by torch library, since it is in a boxed fashion
PYBIND11_MODULE(c_operators, m) {
  m.def("sum_dim", &flag_gems::sum_dim);
  m.def("add_tensor", &flag_gems::add_tensor);
  m.def("rms_norm", &flag_gems::rms_norm);
  m.def("fused_add_rms_norm", &flag_gems::fused_add_rms_norm);
  m.def("nonzero", &flag_gems::nonzero);
  // Rotary embedding
  m.def("rotary_embedding", &flag_gems::rotary_embedding);
  m.def("rotary_embedding_inplace", &flag_gems::rotary_embedding_inplace);
  m.def("bmm", &flag_gems::bmm);
  // div
  m.def("div.Tensor", &flag_gems::true_div);
  m.def("div_.Tensor", &flag_gems::true_div_);
  m.def("div.Tensor_mode", &flag_gems::div_mode);
  m.def("div_.Tensor_mode", &flag_gems::div_mode_);
  m.def("div.Scalar", &flag_gems::true_div);
  m.def("div_.Scalar", &flag_gems::true_div_);
  m.def("div.Scalar_mode", &flag_gems::div_mode);
  m.def("div_.Scalar_mode", &flag_gems::div_mode_);
  m.def("floor_divide", &flag_gems::floor_div);
  m.def("floor_divide_.Tensor", &flag_gems::floor_div_);
  m.def("floor_divide.Scalar", &flag_gems::floor_div);
  m.def("floor_divide_.Scalar", &flag_gems::floor_div_);
  m.def("divide.Tensor", &flag_gems::true_div);
  m.def("divide_.Tensor", &flag_gems::true_div_);
  m.def("divide.Scalar", &flag_gems::true_div);
  m.def("divide_.Scalar", &flag_gems::true_div_);
  m.def("divide.Tensor_mode", &flag_gems::div_mode);
  m.def("divide_.Tensor_mode", &flag_gems::div_mode_);
  m.def("divide.Scalar_mode", &flag_gems::div_mode);
  m.def("divide_.Scalar_mode", &flag_gems::div_mode_);
  m.def("true_divide.Tensor", &flag_gems::true_div);
  m.def("true_divide_.Tensor", &flag_gems::true_div_);
  m.def("remainder.Scalar", &flag_gems::remainder);
  m.def("remainder_.Scalar", &flag_gems::remainder_);
  m.def("remainder.Tensor", &flag_gems::remainder);
  m.def("remainder_.Tensor", &flag_gems::remainder_);
  m.def("remainder.Scalar_Tensor", &flag_gems::remainder);
}

namespace flag_gems {
TORCH_LIBRARY(flag_gems, m) {
  m.def(
      "zeros(SymInt[] size, ScalarType? dtype=None,Layout? layout=None, Device? device=None, bool? "
      "pin_memory=None) -> Tensor");
  m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor", {at::Tag::pt2_compliant_tag});
  // Norm
  m.def("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()");
  m.def("addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
  m.def("nonzero(Tensor self) -> Tensor");
  // rotary_embedding
  m.def(
      "rotary_embedding_inplace(Tensor! q, Tensor! k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> ()");
  m.def(
      "rotary_embedding(Tensor q, Tensor k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> (Tensor, Tensor)");  // q and k may be view to other size
  m.def("topk(Tensor x, SymInt k, int dim, bool largest, bool sorted) -> (Tensor, Tensor)");
  m.def("cat(Tensor[] tensors, int dim=0) -> Tensor");
  m.def("bmm(Tensor self, Tensor mat2) -> Tensor");
  m.def(
      "embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool "
      "sparse=False) -> Tensor");
  m.def(
      "embedding_backward(Tensor grad_outputs, Tensor indices, SymInt num_weights, SymInt padding_idx, bool "
      "scale_grad_by_freq, bool sparse) -> Tensor");
  m.def("argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor");
  // div
  m.def("div.Tensor(Tensor self, Tensor other) -> Tensor");
  m.def("div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  m.def("div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor");
  m.def("div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)");
  m.def("div.Scalar(Tensor self, Scalar other) -> Tensor");
  m.def("div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
  m.def("div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor");
  m.def("div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)");
  m.def("floor_divide(Tensor self, Tensor other) -> Tensor");
  m.def("floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  m.def("floor_divide.Scalar(Tensor self, Scalar other) -> Tensor");
  m.def("floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
  m.def("divide.Tensor(Tensor self, Tensor other) -> Tensor");
  m.def("divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  m.def("divide.Scalar(Tensor self, Scalar other) -> Tensor");
  m.def("divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
  m.def("divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor");
  m.def("divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)");
  m.def("divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor");
  m.def("divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)");
  m.def("true_divide.Tensor(Tensor self, Tensor other) -> Tensor");
  m.def("true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  m.def("true_divide.Scalar(Tensor self, Scalar other) -> Tensor");
  m.def("true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
  m.def("remainder.Scalar(Tensor self, Scalar other) -> Tensor");
  m.def("remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)");
  m.def("remainder.Tensor(Tensor self, Tensor other) -> Tensor");
  m.def("remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  m.def("remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(flag_gems, CUDA, m) {
  m.impl("zeros", TORCH_FN(zeros));
  m.impl("sum.dim_IntList", TORCH_FN(sum_dim));
  m.impl("add_tensor", TORCH_FN(add_tensor));
  // Norm
  m.impl("rms_norm", TORCH_FN(rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(fused_add_rms_norm));
  m.impl("addmm", TORCH_FN(addmm));
  m.impl("nonzero", TORCH_FN(nonzero));
  // Rotary embedding
  m.impl("rotary_embedding", TORCH_FN(rotary_embedding));
  m.impl("rotary_embedding_inplace", TORCH_FN(rotary_embedding_inplace));
  m.impl("topk", TORCH_FN(topk));
  m.impl("cat", TORCH_FN(cat));
  m.impl("bmm", TORCH_FN(bmm));
  m.impl("embedding", TORCH_FN(embedding));
  m.impl("embedding_backward", TORCH_FN(embedding_backward));
  m.impl("argmax", TORCH_FN(argmax));
  // div
  m.impl("div.Tensor", TORCH_FN(true_div));
  m.impl("div_.Tensor", TORCH_FN(true_div_));
  m.impl("div.Tensor_mode", TORCH_FN(div_mode));
  m.impl("div_.Tensor_mode", TORCH_FN(div_mode_));
  m.impl("div.Scalar", TORCH_FN(true_div));
  m.impl("div_.Scalar", TORCH_FN(true_div_));
  m.impl("div.Scalar_mode", TORCH_FN(div_mode));
  m.impl("div_.Scalar_mode", TORCH_FN(div_mode_));
  m.impl("floor_divide", TORCH_FN(floor_div));
  m.impl("floor_divide_.Tensor", TORCH_FN(floor_div_));
  m.impl("floor_divide.Scalar", TORCH_FN(floor_div));
  m.impl("floor_divide_.Scalar", TORCH_FN(floor_div_));
  m.impl("divide.Tensor", TORCH_FN(true_div));
  m.impl("divide_.Tensor", TORCH_FN(true_div_));
  m.impl("divide.Scalar", TORCH_FN(true_div));
  m.impl("divide_.Scalar", TORCH_FN(true_div_));
  m.impl("divide.Tensor_mode", TORCH_FN(div_mode));
  m.impl("divide_.Tensor_mode", TORCH_FN(div_mode_));
  m.impl("divide.Scalar_mode", TORCH_FN(div_mode));
  m.impl("divide_.Scalar_mode", TORCH_FN(div_mode_));
  m.impl("true_divide.Tensor", TORCH_FN(true_div));
  m.impl("true_divide_.Tensor", TORCH_FN(true_div_));
  m.impl("remainder.Scalar", TORCH_FN(remainder));
  m.impl("remainder_.Scalar", TORCH_FN(remainder_));
  m.impl("remainder.Tensor", TORCH_FN(remainder));
  m.impl("remainder_.Tensor", TORCH_FN(remainder_));
  m.impl("remainder.Scalar_Tensor", TORCH_FN(remainder));
}
}  // namespace flag_gems
