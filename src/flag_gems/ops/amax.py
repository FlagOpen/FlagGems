import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def amax_func(inp, dim : tl.constexpr, keepdim : tl.constexpr):
    return tl.amax(inp,dim,keepdim)

def amax(inp, dim=None, keepdim=False):
    logging.debug("GEMS AMAX")
    if dim is not None and len(dim) != 0:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
    return unwrap(amax_func[(1,)](inp, dim=dim, keepdim=keepdim))
