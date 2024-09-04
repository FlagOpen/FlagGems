import torch

from .fused import *  # noqa: F403
from .ops import *  # noqa: F403

__version__ = "2.0"

aten_lib = torch.library.Library("aten", "IMPL")


def enable(lib=aten_lib):
    lib.impl("abs", abs, "CUDA")
    lib.impl("add.Tensor", add, "CUDA")
    lib.impl("addmm", addmm, "CUDA")
    lib.impl("bitwise_and.Tensor", bitwise_and_tensor, "CUDA")
    lib.impl("bitwise_and.Scalar", bitwise_and_scalar, "CUDA")
    lib.impl("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor, "CUDA")
    lib.impl("bitwise_not", bitwise_not, "CUDA")
    lib.impl("bitwise_or.Tensor", bitwise_or_tensor, "CUDA")
    lib.impl("bitwise_or.Scalar", bitwise_or_scalar, "CUDA")
    lib.impl("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor, "CUDA")
    lib.impl("bmm", bmm, "CUDA")
    lib.impl("clamp", clamp, "CUDA")
    lib.impl("clamp.Tensor", clamp_tensor, "CUDA")
    lib.impl("cos", cos, "CUDA")
    lib.impl("pad", pad, "CUDA")
    lib.impl("cumsum", cumsum, "CUDA")
    lib.impl("div.Tensor", true_divide, "CUDA")
    lib.impl("div.Scalar", true_divide, "CUDA")
    lib.impl("div.Tensor_mode", div_mode, "CUDA")
    lib.impl("div.Scalar_mode", div_mode, "CUDA")
    lib.impl("divide.Tensor", true_divide, "CUDA")  # divide, an alias for div
    lib.impl("divide.Scalar", true_divide, "CUDA")
    lib.impl("divide.Tensor_mode", div_mode, "CUDA")
    lib.impl("divide.Scalar_mode", div_mode, "CUDA")
    lib.impl("true_divide.Tensor", true_divide, "CUDA")  # true_divide, an alias for div
    lib.impl("true_divide.Scalar", true_divide, "CUDA")
    lib.impl("floor_divide", floor_divide, "CUDA")
    lib.impl("floor_divide.Scalar", floor_divide, "CUDA")
    lib.impl("native_dropout", native_dropout, "AutogradCUDA")
    lib.impl("erf", erf, "CUDA")
    lib.impl("embedding", embedding, "AutogradCUDA")
    lib.impl("eq.Tensor", eq, "CUDA")
    lib.impl("eq.Scalar", eq_scalar, "CUDA")
    lib.impl("exp", exp, "CUDA")
    lib.impl("exponential_", exponential_, "CUDA")
    lib.impl("ge.Tensor", ge, "CUDA")
    lib.impl("ge.Scalar", ge_scalar, "CUDA")
    lib.impl("gelu", gelu, "AutogradCUDA")
    lib.impl("native_group_norm", group_norm, "AutogradCUDA")
    lib.impl("gt.Tensor", gt, "CUDA")
    lib.impl("gt.Scalar", gt_scalar, "CUDA")
    lib.impl("isfinite", isfinite, "CUDA")
    lib.impl("isinf", isinf, "CUDA")
    lib.impl("isnan", isnan, "CUDA")
    lib.impl("minimum", minimum, "CUDA")
    lib.impl("maximum", maximum, "CUDA")
    lib.impl("native_layer_norm", layer_norm, "AutogradCUDA")
    lib.impl("le.Tensor", le, "CUDA")
    lib.impl("le.Scalar", le_scalar, "CUDA")
    lib.impl("lt.Tensor", lt, "CUDA")
    lib.impl("lt.Scalar", lt_scalar, "CUDA")
    lib.impl("rms_norm", rms_norm, "CUDA")
    lib.impl("rand", rand, "CUDA")
    lib.impl("randn", randn, "CUDA")
    lib.impl("rand_like", rand_like, "CUDA")
    lib.impl("randn_like", randn_like, "CUDA")
    lib.impl("zeros", zeros, "CUDA")
    lib.impl("ones", ones, "CUDA")
    lib.impl("full", full, "CUDA")
    lib.impl("zeros_like", zeros_like, "CUDA")
    lib.impl("ones_like", ones_like, "CUDA")
    lib.impl("full_like", full_like, "CUDA")
    lib.impl("resolve_neg", resolve_neg, "CUDA")
    lib.impl("resolve_conj", resolve_conj, "CUDA")
    lib.impl("normal.Tensor_float", normal_tensor_float, "CUDA")
    lib.impl("normal.float_Tensor", normal_float_tensor, "CUDA")
    lib.impl("normal.Tensor_Tensor", normal_tensor_tensor, "CUDA")
    lib.impl("normal.float_float", normal_float_float, "CUDA")
    lib.impl("uniform_", uniform_, "CUDA")
    lib.impl("mean", mean, "CUDA")
    lib.impl("mean.dim", mean_dim, "CUDA")
    lib.impl("mm", mm, "CUDA")
    lib.impl("mul.Tensor", mul, "CUDA")
    lib.impl("multinomial", multinomial, "CUDA")
    lib.impl("mv", mv, "CUDA")
    lib.impl("ne.Tensor", ne, "CUDA")
    lib.impl("ne.Scalar", ne_scalar, "CUDA")
    lib.impl("neg", neg, "CUDA")
    lib.impl("pow.Scalar", pow_scalar, "CUDA")
    lib.impl("pow.Tensor_Scalar", pow_tensor_scalar, "CUDA")
    lib.impl("pow.Tensor_Tensor", pow_tensor_tensor, "CUDA")
    lib.impl("reciprocal", reciprocal, "CUDA")
    lib.impl("relu", relu, "AutogradCUDA")
    lib.impl("rsqrt", rsqrt, "CUDA")
    lib.impl("sigmoid", sigmoid, "AutogradCUDA")
    lib.impl("silu", silu, "AutogradCUDA")
    lib.impl("sin", sin, "CUDA")
    lib.impl("softmax.int", softmax, "AutogradCUDA")
    lib.impl("sub.Tensor", sub, "CUDA")
    lib.impl("tanh", tanh, "AutogradCUDA")
    lib.impl("triu", triu, "CUDA")
    lib.impl("topk", topk, "CUDA")
    lib.impl("var_mean.correction", var_mean, "CUDA")
    lib.impl("linalg_vector_norm", vector_norm, "CUDA")
    lib.impl("where.self", where_self, "CUDA")
    lib.impl("where.ScalarSelf", where_scalar_self, "CUDA")
    lib.impl("where.ScalarOther", where_scalar_other, "CUDA")
    lib.impl("max", max, "CUDA")
    lib.impl("max.dim", max_dim, "CUDA")
    lib.impl("min", min, "CUDA")
    lib.impl("min.dim", min_dim, "CUDA")
    lib.impl("amax", amax, "CUDA")
    lib.impl("argmax", argmax, "CUDA")
    lib.impl("prod", prod, "CUDA")
    lib.impl("prod.dim_int", prod_dim, "CUDA")
    lib.impl("sum", sum, "CUDA")
    lib.impl("sum.dim_IntList", sum_dim, "CUDA")
    lib.impl("all", all, "CUDA")
    lib.impl("all.dim", all_dim, "CUDA")
    lib.impl("all.dims", all_dims, "CUDA")
    lib.impl("any", any, "CUDA")
    lib.impl("any.dim", any_dim, "CUDA")
    lib.impl("any.dims", any_dims, "CUDA")
    lib.impl("log_softmax.int", log_softmax, "AutogradCUDA")
    lib.impl("outer", outer, "AutogradCUDA")
    lib.impl("cross_entropy_loss", cross_entropy_loss, "AutogradCUDA")
    # lib.impl("scatter.src", scatter_src, "CUDA")
    # lib.impl("scatter.reduce", scatter_reduce, "CUDA")
    # lib.impl("gather", gather, "CUDA")
    # lib.impl("gather.out", gather_out, "CUDA")
    lib.impl("isclose", isclose, "CUDA")
    lib.impl("allclose", allclose, "CUDA")
    lib.impl("flip", flip, "CUDA")
    lib.impl("tile", tile, "CUDA")
    lib.impl("index_select", index_select, "CUDA")
    lib.impl("masked_fill", masked_fill, "CUDA")
    lib.impl("_unique2", _unique2, "CUDA")
    lib.impl("nonzero", nonzero, "CUDA")
    lib.impl("repeat", repeat, "CUDA")


class use_gems:
    def __init__(self):
        self.lib = torch.library.Library("aten", "IMPL")

    def __enter__(self):
        enable(self.lib)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lib


__all__ = [
    "enable",
    "use_gems",
]
