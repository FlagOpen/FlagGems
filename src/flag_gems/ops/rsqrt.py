import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def rsqrt_func(x):
    return 1.0 / tl.sqrt(x.to(tl.float32))


def rsqrt(A):
    print("GEMS RSQRT")
    return rsqrt_func(A)


def rsqrt_(A):
    print("GEMS RSQRT_")
    return rsqrt_func(A, out0=A)
