import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def mean_kernel(inp, out_type: tl.constexpr):
    out = tl.mean(inp)
    return out.to(out_type)


def mean(inp, *, dtype=None):
    logging.debug("GEMS MEAN")
    if dtype is None:
        dtype = inp.dtype
    dtype = type_convert(dtype)
    out = unwrap(mean_kernel[(1,)](inp, dtype))
    return out


@triton.jit
def mean_dim_kernel(inp, dim: tl.constexpr, keepdim: tl.constexpr, out_type: tl.constexpr):
    out = tl.mean(inp, axis=dim, keep_dims=keepdim)
    return out.to(out_type)

def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logging.debug("GEMS MEAN DIM")

    if dtype is None:
        dtype = x.dtype
    dtype = type_convert(dtype)
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = unwrap(mean_dim_kernel[(1,)](x, None, keepdim, dtype))
        return out

    dim = [d % x.ndim for d in dim]
    out = unwrap(mean_dim_kernel[(1,)](x, dim, keepdim, dtype))
    return out

def type_convert(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.bfloat16:
        return tl.bfloat16
