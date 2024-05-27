import triton
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def bitwise_or_func(x, y):
    return x | y


def bitwise_or_tensor(A, B):
    logging.debug("GEMS BITWISE OR")
    O = bitwise_or_func(A, B)
    return O


@pointwise_dynamic(is_tensor=[True, False])
@triton.jit
def bitwise_or_func_scalar(x, y):
    return x | y


def bitwise_or_scalar(A, B):
    logging.debug("GEMS BITWISE OR SCALAR")
    O = bitwise_or_func_scalar(A, B)
    return O


def bitwise_or_scalar_tensor(A, B):
    logging.debug("GEMS BITWISE OR SCALAR TENSOR")
    O = bitwise_or_func_scalar(B, A)
    return O
