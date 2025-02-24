import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func(x, y):
    return x < y


def lt(A, B):
    logging.debug("GEMS_CAMBRICON LT")
    return lt_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func_scalar(x, y):
    return x < y


def lt_scalar(A, B):
    logging.debug("GEMS_CAMBRICON LT SCALAR")
    return lt_func_scalar(A, B)
