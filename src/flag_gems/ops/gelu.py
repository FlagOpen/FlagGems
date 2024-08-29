import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic

try:
    from triton.language.extra.mlu.libdevice import fast_erf, pow, fast_tanh
except ImportError:
    try:
        from triton.language.math import erf, pow, tanh
    except ImportError:
        from triton.language.libdevice import fast_erf, pow, fast_tanh


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    scale: tl.constexpr = 0.7071067811
    output = 0.5 * x * (1 + fast_erf(x * scale))
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    x_f32 = x.to(tl.float32)
    output = (
        0.5 * x * (1 + fast_tanh(x * 0.79788456 * (1 + 0.044715 * x_f32 * x_f32)))
    )
    return output


def gelu(A, *, approximate="none"):
    logging.debug("GEMS GELU")
    if approximate == "tanh":
        return gelu_tanh(A)
    else:
        return gelu_none(A)
