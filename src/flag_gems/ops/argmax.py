import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def argmax_func(inp, dim: tl.constexpr, keep_dim: tl.constexpr):
    out_index = tl.argmax_(inp, axis=dim, keep_dims=keep_dim)
    return out_index

def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logging.debug("GEMS ARGMAX")
    if dim is None:
        return unwrap(argmax_func[(1,)](inp, keepdim))
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        inp = inp.contiguous()
        return unwrap(argmax_func[(1,)](inp, dim, keepdim))
