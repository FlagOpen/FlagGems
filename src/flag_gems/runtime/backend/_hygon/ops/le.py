import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def le_func(x, y):
    return x.to(tl.float32) <= y


def le(A, B):
    logging.debug("GEMS LE")
    return le_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def le_func_scalar(x, y):
    return x.to(tl.float32) <= y


def le_scalar(A, B):
    logging.debug("GEMS LE SCALAR")
    return le_func_scalar(A, B)
