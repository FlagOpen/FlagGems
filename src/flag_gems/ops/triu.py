import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def triu_func(x, diagonal: tl.constexpr):
    return tl.triu(x, diagonal)

def triu(A, diagonal=0):
    logging.debug("GEMS TRIU")
    A = A.contiguous()
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    return unwrap(triu_func[(1,)](A, diagonal))