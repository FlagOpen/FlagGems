import logging

import triton

from ..utils import unwrap


@triton.jit
def neg_func(x):
    return (0 - x)


def neg(A):
    logging.debug("GEMS NEG")
    return unwrap(neg_func[(1,)](A))
