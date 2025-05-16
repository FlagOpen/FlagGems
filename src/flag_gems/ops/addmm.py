import logging
import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_func(input, mat1, mat2, beta:tl.constexpr, alpha:tl.constexpr, out_dtype:tl.constexpr):
    mat1 = mat1.to(tl.float32)
    mat2 = mat2.to(tl.float32)
    input = input.to(tl.float32)
    tmp = tl.dot(mat1, mat2)
    tmp *= alpha
    tmp1 = beta * input
    tmp += tmp1
    return tmp.to(out_dtype)

_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]

def get_higher_dtype(a, b, c):
    if a is b and a is c:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes
    assert c in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a
        if c is d:
            return c

def type_convert(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.bfloat16:
        return tl.bfloat16

def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logging.debug("GEMS ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    dot_out_dtype = get_higher_dtype(bias.dtype, mat1.dtype, mat2.dtype)
    dot_out_dtype = type_convert(dot_out_dtype)
    return unwrap(addmm_func[(1,)](bias, mat1, mat2, beta, alpha, dot_out_dtype))
