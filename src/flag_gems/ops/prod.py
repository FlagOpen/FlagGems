import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def prod_kernel(inp):
    return tl.prod(inp)

@triton.jit
def prod_dim_kernel(inp, dim : tl.constexpr, keepdim : tl.constexpr):
    return tl.prod(inp, dim, keepdim)

def prod(inp, *, dtype=None):
    logging.debug("GEMS PROD")
    return unwrap(prod_kernel[(1,)](inp))

def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logging.debug("GEMS PROD DIM")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    inp = inp.contiguous()

    return unwrap(prod_dim_kernel[(1,)](inp, dim, keepdim))
