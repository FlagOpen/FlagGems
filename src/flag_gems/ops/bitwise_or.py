import logging

import torch
import triton

from ..utils import unwrap


@triton.jit
def bitwise_or_func(x, y):
    r = x | y
    return r.to(x.type.element_ty)

@triton.jit
def bitwise_or_tensor_scalar(x, y):
    r = x | y
    return r.to(x.type.element_ty)

@triton.jit
def bitwise_or_scalar_tensor(x, y):
    r = x | y
    return r.to(y.type.element_ty)

def bitwise_or(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return unwrap(bitwise_or_func[(1,)](x, y))
    elif isinstance(x, torch.Tensor):
        return unwrap(bitwise_or_tensor_scalar[(1,)](x, y))
    elif isinstance(y, torch.Tensor):
        return unwrap(bitwise_or_scalar_tensor[(1,)](x, y))
    else:
        return torch.tensor(x | y)
