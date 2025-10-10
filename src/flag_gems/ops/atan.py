import logging
import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
logger = logging.getLogger(__name__)

@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def atan_kernel(x):
    x = x.to(tl.float32)

    PI_HALF = 1.5707963267948966
    TAN_PI_8 = 0.41421356237309503
    PI_8 = 0.39269908169872414

    sign = tl.where(x < 0, -1.0, 1.0)
    x_abs = tl.abs(x)

    use_identity = x_abs > 1.0
    y = tl.where(use_identity, 1.0 / x_abs, x_abs)

    is_large_y = y > TAN_PI_8
    arg = tl.where(is_large_y, (y - TAN_PI_8) / (1.0 + y * TAN_PI_8), y)

    y2 = arg * arg
    a1 = 0.9999999999999999
    a3 = -0.3333333333333293
    a5 = 0.1999999999907175
    a7 = -0.1428571407255139
    a9 = 0.1111109401133329
    
    atan_arg = arg * (a1 + y2 * (a3 + y2 * (a5 + y2 * (a7 + y2 * a9))))

    atan_y = tl.where(is_large_y, PI_8 + atan_arg, atan_arg)
    result = tl.where(use_identity, PI_HALF - atan_y, atan_y)
    result = result * sign
    return result

def atan(A):
    logger.debug("GEMS ATAN")
    return atan_kernel(A)

def atan_(A):
    logger.debug("GEMS ATAN_")
    atan_kernel(A, out0=A)
    return A