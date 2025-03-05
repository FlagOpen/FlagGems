import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def eq_func(x, y):
    return x.to(tl.float32) == y.to(tl.float32)

def promote_binary_type(x, y):
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    if x.dtype == torch.int64:
        x = x.to(torch.float32)
    if y.dtype == torch.int64:
        y = y.to(torch.float32)
    return x, y

def eq(A, B):
    logging.debug("GEMS EQ")
    A, B = promote_binary_type(A, B)
    return unwrap(eq_func[(1,)](A, B))

@triton.jit
def eq_func_scalar(x, y):
    return x.to(tl.float32) == y.to(tl.float32)

def eq_scalar(A, B):
    logging.debug("GEMS EQ SCALAR")
    A, B = promote_binary_type(A, B)
    return unwrap(eq_func_scalar[(1,)](A, B))
