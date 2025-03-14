import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def sin_func(x):
    return tl.sin(x)

def promote_unary_type(x):
    x = torch.as_tensor(x)
    if x.dtype == torch.int64:
        x = x.to(torch.float32)
    return x

def sin(A):
    logging.debug("GEMS SIN")
    A = promote_unary_type(A)
    return unwrap(sin_func[(1,)](A))
