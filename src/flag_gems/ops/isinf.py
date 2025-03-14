import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def isinf_func(x):
    return x + 1 == x


def isinf(A):
    logging.debug("GEMS ISINF")
    return unwrap(isinf_func[(1,)](A))
