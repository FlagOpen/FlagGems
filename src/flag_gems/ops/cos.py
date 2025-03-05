import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def cos_func(x):
    return tl.cos(x)

def cos(A):
    logging.debug("GEMS COS")
    return unwrap(cos_func[(1,)](A))
