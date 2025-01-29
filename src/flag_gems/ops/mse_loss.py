import logging
import math
from enum import Enum

import torch
import triton
import triton.language as tl

from ..runtime import torch_device_fn
from ..utils import libentry
from ..utils import triton_lang_extension as tle


@libentry()
@triton.jit
def kernel_1(inp, target, mid, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    target_ptrs = target + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    target_val = tl.load(target_ptrs, mask=mask, other=0).to(cdtype)
    sub = inp_val - target_val
    pow_val = sub * sub
    # Reduction.MEAN.value: 1 Reduction.SUM.value: 2
    if reduction == 1:
        sum_val = tl.sum(pow_val) / M
    else:
        sum_val = tl.sum(pow_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def mse_loss(inp, target, reduction=Reduction.MEAN.value):
    logging.debug("GEMS MSE LOSS")
    if reduction == Reduction.NONE.value:
        return torch.pow(inp - target, 2)

    M = inp.numel()
    dtype = inp.dtype
    if dtype is torch.bool:
        inp = inp.to(torch.int64)
        dtype = torch.int64

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        kernel_1[(mid_size, 1, 1)](inp, target, mid, M, block_size, reduction)
        kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out
