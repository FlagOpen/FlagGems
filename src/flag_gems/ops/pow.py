import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def pow_func(x, exponent):
    out = tl.pow(x.to(tl.float32), exponent.to(tl.float32))
    return out.to(x.type.element_ty)

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

def pow_tensor_tensor(A, exponent):
    logging.debug("GEMS POW_TENSOR_TENSOR")
    A, exponent, output_dtype = promote_binary_types(A, exponent)
    out = unwrap(pow_func[(1,)](A, exponent))
    if output_dtype is not None:
        out = out.to(output_dtype)
    return out


@triton.jit
def pow_func_tensor_scalar(x, exponent):
    out = tl.pow(x.to(tl.float32), exponent.to(tl.float32))
    return out.to(x.type.element_ty)


def pow_tensor_scalar(A, exponent):
    logging.debug("GEMS POW_TENSOR_SCALAR")
    A, exponent, output_dtype = promote_binary_types(A, exponent)
    out = unwrap(pow_func_tensor_scalar[(1,)](A, exponent))
    if output_dtype is not None:
        out = out.to(output_dtype)
    return out


@triton.jit
def pow_func_scalar_tensor(x, exponent):
    out = tl.pow(x.to(tl.float32), exponent.to(tl.float32))
    return out.to(exponent.type.element_ty)


def pow_scalar(A, exponent):
    logging.debug("GEMS POW_SCALAR")
    A, exponent, output_dtype = promote_binary_types(A, exponent)
    out = unwrap(pow_func_scalar_tensor[(1,)](A, exponent))
    if output_dtype is not None:
        out = out.to(output_dtype)
    return out

