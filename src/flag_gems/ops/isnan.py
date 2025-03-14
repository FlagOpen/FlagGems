import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def isnan_func(x):
    return x != x


def isnan(A):
    logging.debug("GEMS ISNAN")
    return unwrap(isnan_func[(1,)](A))
