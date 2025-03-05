import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def sum_func(inp):
    return tl.sum_(inp)

def sum(inp, *, dtype=None):
    logging.debug("GEMS SUM")
    out = unwrap(sum_func[(1,)](inp))
    return out

@triton.jit
def sum_func_dim(inp, dim: tl.constexpr, keepdim: tl.constexpr):
    out = tl.sum_(inp, axis=dim, keep_dims=keepdim)
    return out

def sum_dim(inp, dim, keepdim=False, *, dtype=None):
    logging.debug("GEMS SUM DIM")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    if dim == []:
        if not keepdim:
            return sum(inp, dtype=dtype)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum(inp, dtype=dtype), [1] * dim_num)

    out = unwrap(sum_func_dim[(1,)](inp, dim, keepdim))
    return out
