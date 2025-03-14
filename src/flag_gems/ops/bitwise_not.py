import logging
  
import triton

from ..utils import unwrap


@triton.jit
def bitwise_not_func(x):
    return ~x


def bitwise_not(A):
    logging.debug("GEMS BITWISE NOT")
    return unwrap(bitwise_not_func[(1,)](A))
