#include <pybind11/pybind11.h>
#include "flag_gems/operators.h"

// TODO: use pytorch's argparse utilities to generate CPython bindings, since it is more efficient than
// bindings provided by torch library, since it is in a boxed fashion
PYBIND11_MODULE(c_operators, m) {
}

namespace flag_gems {
TORCH_LIBRARY(flag_gems, m) {
  m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor", {at::Tag::pt2_compliant_tag});
  // Norm
  m.def("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()");
  m.def("nonzero(Tensor self) -> Tensor");
  // rotary_embedding
  m.def(
      "rotary_embedding_inplace(Tensor! q, Tensor! k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> ()");
  m.def(
      "rotary_embedding(Tensor q, Tensor k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> (Tensor, Tensor)");  // q and k may be view to other size
  m.def("bmm(Tensor self, Tensor mat2) -> Tensor");
  // fill operator declaration 
  m.def("fill.Scalar(Tensor self, Scalar value) -> Tensor");
  m.def("fill.Tensor(Tensor self, Tensor value) -> Tensor");
  m.def("fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)");
  m.def("fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(flag_gems, CUDA, m) {
  m.impl("sum.dim_IntList", TORCH_FN(sum_dim));
  m.impl("add_tensor", TORCH_FN(add_tensor));
  // Norm
  m.impl("rms_norm", TORCH_FN(rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(fused_add_rms_norm));
  m.impl("nonzero", TORCH_FN(nonzero));
  // Rotary embedding
  m.impl("rotary_embedding", TORCH_FN(rotary_embedding));
  m.impl("rotary_embedding_inplace", TORCH_FN(rotary_embedding_inplace));
  m.impl("bmm", TORCH_FN(bmm));
  // Fill operator binding
  m.impl("fill.Scalar", TORCH_FN(flag_gems::fill_scalar));
  m.impl("fill.Tensor", TORCH_FN(flag_gems::fill_tensor));
  m.impl("fill_.Scalar", TORCH_FN(flag_gems::fill_scalar_));
  m.impl("fill_.Tensor", TORCH_FN(flag_gems::fill_tensor_));
}
}  // namespace flag_gems
