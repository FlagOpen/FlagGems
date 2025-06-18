import logging

import triton

from ..utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def le_func(x, y):
    return x <= y


def le(A, B):
    logging.debug("GEMS_CAMBRICON LE")
    return le_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def le_func_scalar(x, y):
    return x <= y


def le_scalar(A, B):
    logging.debug("GEMS_CAMBRICON LE SCALAR")
    return le_func_scalar(A, B)
