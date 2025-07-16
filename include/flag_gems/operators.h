#pragma once
#include <optional>
#include "torch/torch.h"

namespace flag_gems {
at::Tensor zeros(at::IntArrayRef size,
                 c10::optional<at::ScalarType> dtype = ::std::nullopt,
                 c10::optional<at::Layout> layout = ::std::nullopt,
                 c10::optional<at::Device> device = ::std::nullopt,
                 c10::optional<bool> pin_memory = ::std::nullopt);
at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_, const c10::Scalar &alpha);
at::Tensor mm(const at::Tensor &mat1, const at::Tensor &mat2);
at::Tensor sum_dim(const at::Tensor &self,
                   at::OptionalIntArrayRef dim,
                   bool keepdim = false,
                   ::std::optional<at::ScalarType> dtype = ::std::nullopt);

at::Tensor rms_norm(const at::Tensor &input, const at::Tensor &weight, double epsilon = 1e-5);
void fused_add_rms_norm(at::Tensor &input,
                        at::Tensor &residual,
                        const at::Tensor &weight,
                        double epsilon = 1e-5);
at::Tensor addmm(const at::Tensor &self,
                 const at::Tensor &mat1,
                 const at::Tensor &mat2,
                 const at::Scalar &beta = 1.0,
                 const at::Scalar &alpha = 1.0);
at::Tensor nonzero(const at::Tensor &inp);
// Rotary embedding
void rotary_embedding_inplace(at::Tensor &q,
                              at::Tensor &k,
                              const at::Tensor &cos,
                              const at::Tensor &sin,
                              const std::optional<at::Tensor> &position_ids = std::nullopt,
                              bool rotary_interleaved = false);
std::tuple<at::Tensor, at::Tensor> rotary_embedding(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const std::optional<at::Tensor> &position_ids = std::nullopt,
    bool rotary_interleaved = false);

at::Tensor cat(const at::TensorList &tensors, int64_t dim = 0);
at::Tensor bmm(const at::Tensor &A, const at::Tensor &B);
}  // namespace flag_gems
