import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def ne_func(x, y):
    return x.to(tl.float32) != y.to(tl.float32)

def ne(A, B):
    logging.debug("GEMS NE")
    return unwrap(ne_func[(1,)](A, B))

def ne_scalar(A, B):
    logging.debug("GEMS NE SCALAR")
    return unwrap(ne_func[(1,)](A, B))
