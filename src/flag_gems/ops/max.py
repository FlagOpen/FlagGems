import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def max_func(inp):
    return tl.max_(inp)

@triton.jit
def max_func_dim(inp, dim: tl.constexpr, keepdim: tl.constexpr):
    values, indices = tl.max_(inp, axis=dim, keep_dims=keepdim)
    return values, indices

def max(inp):
    logging.debug("GEMS MAX")
    return unwrap(max_func[(1,)](inp))

def max_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    inp = inp.contiguous()
    values, indices = unwrap(max_func_dim[(1,)](inp, dim, keepdim))
    return values, indices
