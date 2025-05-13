import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def fill_scalar_kernel(
    input,
    zero,
    value,
    outdtype : tl.constexpr
):
    out = zero + value
    out = out.to(outdtype)
    return out

@triton.jit
def fill_tensor_kernel(
    input,
    zero,
    value,
    outdtype : tl.constexpr
):
    out = zero + value
    out = out.to(outdtype)
    return out


def type_convert(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.bfloat16:
        return tl.bfloat16


def fill_tensor(input, value):
    logging.debug("GEMS FILL")
    zero_tensor = torch.zeros(input.shape, dtype=input.dtype, device=input.device)
    return unwrap(fill_tensor_kernel[(1,)](input, zero_tensor, value, type_convert(input.dtype)))


def fill_scalar(input, value):
    logging.debug("GEMS FILL")
    zero_tensor = torch.zeros(input.shape, dtype=input.dtype, device=input.device)
    value_tensor = torch.tensor(value, dtype=input.dtype, device=input.device)
    return unwrap(fill_scalar_kernel[(1,)](input, zero_tensor, value_tensor.item(), type_convert(input.dtype)))
