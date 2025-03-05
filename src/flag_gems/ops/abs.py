import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def abs_func(x):
    return tl.abs(x)

def promote_unary_type(x):
    x = torch.as_tensor(x)
    output_dtype = None
    if x.dtype == torch.int64:
        output_dtype = x.dtype
        x = x.to(torch.float32)
    return x, output_dtype

def abs(A):
    logging.debug("GEMS ABS")
    A, output_dtype = promote_unary_type(A)
    out = unwrap(abs_func[(1,)](A))
    if output_dtype is not None:
        out = out.to(output_dtype)
    return out
