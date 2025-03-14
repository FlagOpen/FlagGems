import logging

import torch
import triton

from ..utils import unwrap

@triton.jit
def sub_func(x, y, alpha):
    out = x - y * alpha
    return out.to(x.type.element_ty)


@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    out = x - y * alpha
    return out.to(x.type.element_ty)


@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    out = x - y * alpha
    return out.to(y.type.element_ty)


def sub(A, B, *, alpha=1):
    logging.debug("GEMS SUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return unwrap(sub_func[(1,)](A, B, alpha))
    elif isinstance(A, torch.Tensor):
        return unwrap(sub_func_tensor_scalar[(1,)](A, B, alpha))
    elif isinstance(B, torch.Tensor):
        return unwrap(sub_func_scalar_tensor[(1,)](A, B, alpha))
    else:
        # Both scalar
        return A - B * alpha
