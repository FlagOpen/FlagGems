import torch

from .fused import *  # noqa: F403
from .ops import *  # noqa: F403

__version__ = "2.0"

aten_lib = torch.library.Library("aten", "IMPL")


def enable(lib=aten_lib):
    lib.impl("abs", abs, "PrivateUse1")
    lib.impl("add.Tensor", add, "PrivateUse1")
    lib.impl("addmm", addmm, "PrivateUse1")
    lib.impl("bitwise_and.Tensor", bitwise_and_tensor, "PrivateUse1")
    lib.impl("bitwise_and.Scalar", bitwise_and_scalar, "PrivateUse1")
    lib.impl("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor, "PrivateUse1")
    lib.impl("bitwise_not", bitwise_not, "PrivateUse1")
    lib.impl("bitwise_or.Tensor", bitwise_or_tensor, "PrivateUse1")
    lib.impl("bitwise_or.Scalar", bitwise_or_scalar, "PrivateUse1")
    lib.impl("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor, "PrivateUse1")
    lib.impl("bmm", bmm, "PrivateUse1")
    lib.impl("clamp", clamp, "PrivateUse1")
    lib.impl("clamp.Tensor", clamp_tensor, "PrivateUse1")
    lib.impl("cos", cos, "PrivateUse1")
    lib.impl("cumsum", cumsum, "PrivateUse1")
    lib.impl("div.Tensor", div, "PrivateUse1")
    lib.impl("native_dropout", native_dropout, "AutogradPrivateUse1")
    lib.impl("eq.Tensor", eq, "PrivateUse1")
    lib.impl("eq.Scalar", eq_scalar, "PrivateUse1")
    lib.impl("exp", exp, "PrivateUse1")
    lib.impl("ge.Tensor", ge, "PrivateUse1")
    lib.impl("ge.Scalar", ge_scalar, "PrivateUse1")
    lib.impl("gelu", gelu, "PrivateUse1")
    lib.impl("native_group_norm", group_norm, "AutogradPrivateUse1")
    lib.impl("gt.Tensor", gt, "PrivateUse1")
    lib.impl("gt.Scalar", gt_scalar, "PrivateUse1")
    lib.impl("isinf", isinf, "PrivateUse1")
    lib.impl("isnan", isnan, "PrivateUse1")
    lib.impl("native_layer_norm", layer_norm, "AutogradPrivateUse1")
    lib.impl("le.Tensor", le, "PrivateUse1")
    lib.impl("le.Scalar", le_scalar, "PrivateUse1")
    lib.impl("lt.Tensor", lt, "PrivateUse1")
    lib.impl("lt.Scalar", lt_scalar, "PrivateUse1")
    lib.impl("rms_norm", rms_norm, "PrivateUse1")
    lib.impl("rand", rand, "PrivateUse1")
    lib.impl("randn", randn, "PrivateUse1")
    lib.impl("rand_like", rand_like, "PrivateUse1")

    lib.impl("mean", mean, "PrivateUse1")
    lib.impl("mean.dim", mean_dim, "PrivateUse1")
    lib.impl("mm", mm, "PrivateUse1")
    lib.impl("mul.Tensor", mul, "PrivateUse1")
    lib.impl("mv", mv, "PrivateUse1")
    lib.impl("ne.Tensor", ne, "PrivateUse1")
    lib.impl("ne.Scalar", ne_scalar, "PrivateUse1")
    lib.impl("neg", neg, "PrivateUse1")
    lib.impl("pow.Scalar", pow_scalar, "PrivateUse1")
    lib.impl("pow.Tensor_Scalar", pow_tensor_scalar, "PrivateUse1")
    lib.impl("pow.Tensor_Tensor", pow_tensor_tensor, "PrivateUse1")
    lib.impl("reciprocal", reciprocal, "PrivateUse1")
    lib.impl("relu", relu, "AutogradPrivateUse1")
    lib.impl("rsqrt", rsqrt, "PrivateUse1")
    lib.impl("sigmoid", sigmoid, "AutogradPrivateUse1")
    lib.impl("silu", silu, "AutogradPrivateUse1")
    lib.impl("sin", sin, "PrivateUse1")
    lib.impl("softmax.int", softmax, "AutogradPrivateUse1")
    lib.impl("sub.Tensor", sub, "PrivateUse1")
    lib.impl("tanh", tanh, "AutogradPrivateUse1")
    lib.impl("triu", triu, "PrivateUse1")
    lib.impl("var_mean.correction", var_mean, "PrivateUse1")
    lib.impl("linalg_vector_norm", vector_norm, "PrivateUse1")
    lib.impl("where.self", where_self, "PrivateUse1")
    lib.impl("where.ScalarSelf", where_scalar_self, "PrivateUse1")
    lib.impl("where.ScalarOther", where_scalar_other, "PrivateUse1")
    lib.impl("max", max, "PrivateUse1")
    lib.impl("max.dim", max_dim, "PrivateUse1")
    lib.impl("min", min, "PrivateUse1")
    lib.impl("min.dim", min_dim, "PrivateUse1")
    lib.impl("amax", amax, "PrivateUse1")
    lib.impl("argmax", argmax, "PrivateUse1")
    lib.impl("prod", prod, "PrivateUse1")
    lib.impl("prod.dim_int", prod_dim, "PrivateUse1")
    lib.impl("sum", sum, "PrivateUse1")
    lib.impl("sum.dim_IntList", sum_dim, "PrivateUse1")
    lib.impl("all", all, "PrivateUse1")
    lib.impl("all.dim", all_dim, "PrivateUse1")
    lib.impl("all.dims", all_dims, "PrivateUse1")
    lib.impl("any", any, "PrivateUse1")
    lib.impl("any.dim", any_dim, "PrivateUse1")
    lib.impl("any.dims", any_dims, "PrivateUse1")
    lib.impl("log_softmax.int", log_softmax, "AutogradPrivateUse1")
    lib.impl("outer", outer, "AutogradPrivateUse1")
    lib.impl("cross_entropy_loss", cross_entropy_loss, "AutogradPrivateUse1")
    lib.impl("isclose", isclose, "PrivateUse1")
    lib.impl("allclose", allclose, "PrivateUse1")


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
