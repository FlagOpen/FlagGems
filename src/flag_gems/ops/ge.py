import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def ge_func(x, y):
    return x >= y

def ge(A, B):
    logging.debug("GEMS GE")
    return unwrap(ge_func[(1,)](A, B))

def ge_scalar(A, B):
    logging.debug("GEMS GE SCALAR")
    return unwrap(ge_func[(1,)](A, B))

