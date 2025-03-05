import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def exp_func(x):
    return tl.exp(x)


def exp(A):
    logging.debug("GEMS EXP")
    return unwrap(exp_func[(1,)](A))
