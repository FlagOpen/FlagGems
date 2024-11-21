import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic

try:
    from triton.language.extra.mlu.libdevice import pow as _pow
except ImportError:
    try:
        from triton.language.math import pow as _pow
    except ImportError:
        from triton.language.libdevice import pow as _pow


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func(x, exponent):
    return _pow(x.to(tl.float32), exponent)


def pow_tensor_tensor(A, exponent):
    logging.debug("GEMS POW_TENSOR_TENSOR")
    return pow_func(A, exponent)

@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_tensor_scalar(x, exponent):
    if exponent.dtype.is_int():
        tmp = x.to(dtype=tl.float32)
        result = tl.full(x.shape, 1, tmp.dtype)
        n = tl.abs(exponent)
        if exponent == 0:
            result = result
        elif n == 1:
            result = tmp
        elif n == 2:
            result = tmp * tmp
        elif n == 3:
            result = tmp * tmp
            result = result * tmp
        elif n == 4:
            result = tmp * tmp
            result = result * result
        else:
            while n > 0:
                if n % 2 == 1:
                    result = result * tmp
                tmp = tmp * x
                n = n // 2
        if exponent < 0:
            result = 1 / result
        return result
    return _pow(x.to(tl.float32), exponent)


def pow_tensor_scalar(A, exponent):
    logging.debug("GEMS POW_TENSOR_SCALAR")
    return pow_func_tensor_scalar(A, exponent)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent)


def pow_scalar(A, exponent):
    logging.debug("GEMS POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)
