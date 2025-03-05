import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def all_kernel_dim(inp, dim: tl.constexpr, keepdim: tl.constexpr):
    return tl.all(inp, axis=dim, keep_dims=keepdim)


def all(inp):
    logging.debug("GEMS ALL")
    out = unwrap(all_kernel_dim[(1,)](inp, None, False))
    return out


def all_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS ALL DIM")
    if dim is None:
        if keepdim:
            out = unwrap(all_kernel_dim[(1,)](inp, None, True))
        else:
            out = unwrap(all_kernel_dim[(1,)](inp, None, False))
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        if not keepdim:
            out = unwrap(all_kernel_dim[(1,)](inp, dim, False))
        else:
            out = unwrap(all_kernel_dim[(1,)](inp, dim, True))
    return out


def all_dims(inp, dim=None, keepdim=False):
    logging.debug("GEMS ALL DIMS")
    if dim is None or isinstance(dim, int):
        return all_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"

    dim = [d % inp.ndim for d in dim]

    out = unwrap(all_kernel_dim[(1,)](inp, dim, keepdim))
    return out
