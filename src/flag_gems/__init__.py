import os
import torch
import triton

from .fused import *  # noqa: F403
from .ops import *  # noqa: F403
from triton.backends.tx8be.driver import CPUDriver

__version__ = "2.0"

aten_lib = torch.library.Library("aten", "IMPL")

IS_TSINGMICRO = os.getenv("GEMS_VENDOR") == "tsingmicro"
FLAGGEMS_DEVICE = os.getenv("FLAGGEMS_DEVICE", "").lower()

if FLAGGEMS_DEVICE == "cpu":
    device = "cpu"
elif FLAGGEMS_DEVICE == "xla":
    device = "xla"
elif FLAGGEMS_DEVICE == "":
    # Maintain original behavior if env var not set
    device = "cpu" if IS_TSINGMICRO else "cuda"
else:
    raise ValueError(
        f"Invalid FLAGGEMS_DEVICE value: {FLAGGEMS_DEVICE}. Must be 'cpu', 'xla', or unset."
    )

if device == "cpu" or device == "xla":
    triton.runtime.driver.set_active(CPUDriver())

def enable(lib=aten_lib):
    if device == "cpu":
        device_key = "CPU"
        device_autograd = "AutogradCPU"
    elif device == "xla":
        device_key = "XLA"
        device_autograd = "AutogradXLA"
    else:  # cuda
        device_key = "CUDA"
        device_autograd = "AutogradCUDA"

    lib.impl("abs", abs, device_key)
    lib.impl("add.Tensor", add, device_key)
    lib.impl("addmm", addmm, device_key)
    lib.impl("bitwise_and.Tensor", bitwise_and_tensor, device_key)
    lib.impl("bitwise_and.Scalar", bitwise_and_scalar, device_key)
    lib.impl("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor, device_key)
    lib.impl("bitwise_not", bitwise_not, device_key)
    lib.impl("bitwise_or.Tensor", bitwise_or, device_key)
    lib.impl("bitwise_or.Scalar", bitwise_or, device_key)
    lib.impl("bitwise_or.Scalar_Tensor", bitwise_or, device_key)
    lib.impl("bmm", bmm, device_key)
    lib.impl("clamp", clamp, device_key)
    lib.impl("clamp.Tensor", clamp_tensor, device_key)
    lib.impl("cos", cos, device_key)
    lib.impl("cumsum", cumsum, device_key)
    lib.impl("div.Tensor", div, device_key)
    lib.impl("native_dropout", native_dropout, device_autograd)
    lib.impl("erf", erf, device_key)
    lib.impl("embedding", embedding, device_autograd)
    lib.impl("eq.Tensor", eq, device_key)
    lib.impl("eq.Scalar", eq_scalar, device_key)
    lib.impl("exp", exp, device_key)
    lib.impl("exponential_", exponential_, device_key)
    lib.impl("ge.Tensor", ge, device_key)
    lib.impl("ge.Scalar", ge_scalar, device_key)
    lib.impl("gelu", gelu, device_key)
    lib.impl("native_group_norm", group_norm, device_autograd)
    lib.impl("gt.Tensor", gt, device_key)
    lib.impl("gt.Scalar", gt_scalar, device_key)
    lib.impl("isfinite", isfinite, device_key)
    lib.impl("isinf", isinf, device_key)
    lib.impl("isnan", isnan, device_key)
    lib.impl("native_layer_norm", layer_norm, device_autograd)
    lib.impl("le.Tensor", le, device_key)
    lib.impl("le.Scalar", le_scalar, device_key)
    lib.impl("lt.Tensor", lt, device_key)
    lib.impl("lt.Scalar", lt_scalar, device_key)
    lib.impl("rms_norm", rms_norm, device_key)
    lib.impl("rand", rand, device_key)
    # lib.impl("randn", randn, device_key)
    lib.impl("rand_like", rand_like, device_key)
    #lib.impl("zeros", zeros, device_key)
    #lib.impl("ones", ones, device_key)
    lib.impl("full", full, device_key)
    # lib.impl("zeros_like", zeros_like, device_key)
    # lib.impl("ones_like", ones_like, device_key)
    lib.impl("full_like", full_like, device_key)
    lib.impl("resolve_neg", resolve_neg, device_key)
    lib.impl("resolve_conj", resolve_conj, device_key)
    lib.impl("normal.Tensor_float", normal_tensor_float, device_key)
    lib.impl("normal.float_Tensor", normal_float_tensor, device_key)
    lib.impl("normal.Tensor_Tensor", normal_tensor_tensor, device_key)
    lib.impl("normal.float_float", normal_float_float, device_key)
    #lib.impl("uniform_", uniform_, device_key)
    lib.impl("mean", mean, device_key)
    lib.impl("mean.dim", mean_dim, device_key)
    lib.impl("mm", mm, device_key)
    lib.impl("mul.Tensor", mul, device_key)
    lib.impl("mv", mv, device_key)
    lib.impl("ne.Tensor", ne, device_key)
    lib.impl("ne.Scalar", ne_scalar, device_key)
    lib.impl("neg", neg, device_key)
    lib.impl("pow.Scalar", pow_scalar, device_key)
    lib.impl("pow.Tensor_Scalar", pow_tensor_scalar, device_key)
    lib.impl("pow.Tensor_Tensor", pow_tensor_tensor, device_key)
    lib.impl("reciprocal", reciprocal, device_key)
    lib.impl("relu", relu, device_autograd)
    lib.impl("rsqrt", rsqrt, device_key)
    lib.impl("sigmoid", sigmoid, device_autograd)
    lib.impl("silu", silu, device_autograd)
    lib.impl("sin", sin, device_key)
    lib.impl("softmax.int", softmax, device_autograd)
    lib.impl("sub.Tensor", sub, device_key)
    lib.impl("tanh", tanh, device_autograd)
    lib.impl("triu", triu, device_key)
    lib.impl("var_mean.correction", var_mean, device_key)
    lib.impl("linalg_vector_norm", vector_norm, device_key)
    lib.impl("where.self", where_self, device_key)
    lib.impl("where.ScalarSelf", where_scalar_self, device_key)
    lib.impl("where.ScalarOther", where_scalar_other, device_key)
    lib.impl("max", max, device_key)
    lib.impl("max.dim", max_dim, device_key)
    lib.impl("min", min, device_key)
    lib.impl("min.dim", min_dim, device_key)
    lib.impl("amax", amax, device_key)
    lib.impl("argmax", argmax, device_key)
    lib.impl("prod", prod, device_key)
    lib.impl("prod.dim_int", prod_dim, device_key)
    lib.impl("sum", sum, device_key)
    lib.impl("sum.dim_IntList", sum_dim, device_key)
    lib.impl("all", all, device_key)
    lib.impl("all.dim", all_dim, device_key)
    lib.impl("all.dims", all_dims, device_key)
    lib.impl("any", any, device_key)
    lib.impl("any.dim", any_dim, device_key)
    lib.impl("any.dims", any_dims, device_key)
    lib.impl("log_softmax.int", log_softmax, device_autograd)
    lib.impl("outer", outer, device_autograd)
    lib.impl("cross_entropy_loss", cross_entropy_loss, device_autograd)
    lib.impl("isclose", isclose, device_key)
    lib.impl("allclose", allclose, device_key)
    lib.impl("flip", flip, device_key)
    lib.impl("fill.Scalar", fill_scalar, device_key)
    lib.impl("fill.Tensor", fill_tensor, device_key)


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
    "device",
]
