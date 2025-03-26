import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def lt_func(x, y):
    return x < y

def lt(A, B):
    logging.debug("GEMS LT")
    return unwrap(lt_func[(1,)](A, B))

def lt_scalar(A, B):
    logging.debug("GEMS LT SCALAR")
    return unwrap(lt_func[(1,)](A, B))
