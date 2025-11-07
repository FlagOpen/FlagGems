import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tan_func(x):
    xf = x.to(tl.float32)
    s = tl.sin(xf)
    c = tl.cos(xf)
    return s / c


def tan(A):
    logger.debug("GEMS TAN")
    return tan_func(A)


def tan_(A):
    logger.debug("GEMS TAN_")
    tan_func(A, out0=A)
    return A
