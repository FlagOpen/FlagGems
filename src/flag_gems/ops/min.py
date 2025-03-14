import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def min_func(inp):
    return tl.min_(inp)


@triton.jit
def min_func_dim(inp, dim: tl.constexpr, keepdim: tl.constexpr):
    return tl.min_(inp, axis=dim, keep_dims=keepdim)


def min(inp):
    logging.debug("GEMS MIN")
    return unwrap(min_func[(1,)](inp))


def min_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS MIN DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    inp = inp.contiguous()
    values, indices = unwrap(min_func_dim[(1,)](inp, dim, keepdim))
    return values, indices
