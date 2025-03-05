import logging

import torch
import triton

from ..utils import unwrap


@triton.jit
def mul_func(x, y):
    out =  x * y
    return out.to(x.type.element_ty)


@triton.jit
def mul_func_scalar(x, y):
    out = x * y
    return out.to(x.type.element_ty)

@triton.jit
def mul_func_scalar_tensor(x, y):
    out = x * y
    return out.to(y.type.element_ty)


def mul(A, B):
    logging.debug("GEMS MUL")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return unwrap(mul_func[(1,)](A, B))
    elif isinstance(A, torch.Tensor):
        return unwrap(mul_func_scalar[(1,)](A, B))
    elif isinstance(B, torch.Tensor):
        return unwrap(mul_func_scalar_tensor[(1,)](A, B))
    else:
        # Both scalar
        return A * B
