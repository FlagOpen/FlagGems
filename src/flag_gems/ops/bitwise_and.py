import logging

import triton
from ..utils import unwrap


@triton.jit
def bitwise_and_func(x, y):
    return x & y


def bitwise_and_tensor(A, B):
    logging.debug("GEMS BITWISE AND")
    return unwrap(bitwise_and_func[(1,)](A, B))


@triton.jit
def bitwise_and_func_scalar(x, y):
    return (x & y).to(x.type.element_ty)


def bitwise_and_scalar(A, B):
    logging.debug("GEMS BITWISE AND SCALAR")
    return unwrap(bitwise_and_func_scalar[(1,)](A, B))


def bitwise_and_scalar_tensor(A, B):
    logging.debug("GEMS BITWISE AND SCALAR TENSOR")
    return unwrap(bitwise_and_func_scalar[(1,)](B, A))
