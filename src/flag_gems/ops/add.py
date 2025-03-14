import logging

import torch
import triton

from ..utils import unwrap


@triton.jit
def add_func(x, y, alpha):
    out =  x + y * alpha
    return out.to(x.type.element_ty)


@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    out =  x + y * alpha
    return out.to(x.type.element_ty)


@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    out =  x + y * alpha
    return out.to(y.type.element_ty)

def promote_binary_types(x, y):
    x_tensor = torch.as_tensor(x)
    y_tensor = torch.as_tensor(y)
    output_dtype = None
    float_dtypes = {torch.float16, torch.bfloat16, torch.float32}

    if x_tensor.dtype == torch.int64 and y_tensor.dtype in float_dtypes:
        output_dtype = y_tensor.dtype
    elif y_tensor.dtype == torch.int64 and x_tensor.dtype in float_dtypes:
        output_dtype = x_tensor.dtype

    if output_dtype is not None:
        x_tensor = x_tensor.to(torch.float32)
        y_tensor = y_tensor.to(torch.float32)

    x = x_tensor if isinstance(x, torch.Tensor) else x_tensor.item()
    y = y_tensor if isinstance(y, torch.Tensor) else y_tensor.item()
    return x, y, output_dtype

def add(A, B, *, alpha=1):
    logging.debug("GEMS ADD")
    A, B, output_dtype = promote_binary_types(A, B)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        out = unwrap(add_func[(1,)](A, B, alpha))
    elif isinstance(A, torch.Tensor):
        out = unwrap(add_func_tensor_scalar[(1,)](A, B, alpha))
    elif isinstance(B, torch.Tensor):
        out = unwrap(add_func_scalar_tensor[(1,)](A, B, alpha))
    else:
        out = torch.tensor(A + B * alpha)
    if output_dtype is not None:
        out = out.to(output_dtype)
    return out
