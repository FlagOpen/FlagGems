import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def rsqrt_func(x):
    return tl.rsqrt(x)


def rsqrt(A):
    logging.debug("GEMS RSQRT")
    return unwrap(rsqrt_func[(1,)](A))
