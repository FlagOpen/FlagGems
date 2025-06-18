import triton

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def neg_func(x):
    return -x


def neg(A):
    print("GEMS NEG")
    return neg_func(A)


def neg_(A):
    print("GEMS NEG_")
    return neg_func(A, out0=A)
