import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def mm_func(A, B, out_type: tl.constexpr):
    out = tl.dot(A, B, out_dtype=out_type)
    return out

_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a

def type_convert(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.bfloat16:
        return tl.bfloat16


def mm(a, b):
    logging.debug("GEMS MM")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    # select dot out type
    dot_out_dtype = get_higher_dtype(a.dtype, b.dtype)
    dot_out_dtype = type_convert(dot_out_dtype)
    # launch kernel
    out = unwrap(mm_func[(1,)](a, b, dot_out_dtype))
    return out
