import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ge_func(x, y):
    return x.to(tl.float32) >= y


def ge(A, B):
    logging.debug("GEMS GE")
    return ge_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def ge_func_scalar(x, y):
    return x.to(tl.float32) >= y


def ge_scalar(A, B):
    logging.debug("GEMS GE SCALAR")
    return ge_func_scalar(A, B)
