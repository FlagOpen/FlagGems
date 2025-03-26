import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def le_func(x, y):
    return x <= y

def le(A, B):
    logging.debug("GEMS LE")
    return unwrap(le_func[(1,)](A, B))

def le_scalar(A, B):
    logging.debug("GEMS LE SCALAR")
    return unwrap(le_func[(1,)](A, B))
