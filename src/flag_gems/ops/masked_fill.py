import logging

import torch
import triton
import triton.language as tl

from ..utils import broadcastable_to, libentry


def cfggen():
    block_m = [1, 2, 4]
    block_n = [1024, 2048, 4096]
    warps = [4, 8, 16]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w)
        for m in block_m
        for n in block_n
        for w in warps
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def masked_fill_kernel(
    inp, expand_mask, value, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    rows_offset = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offset = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = rows_offset < M and cols_offset < N

    offsets = rows_offset * N + cols_offset
    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(not fill_mask) and mask, other=0)
    tl.store(out + offsets, cur_inp, (not fill_mask) and mask)

    cur_val = tl.full((BLOCK_M, BLOCK_N), value, dtype=cur_inp.dtype)
    tl.store(out + offsets, cur_val, fill_mask and mask)


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def masked_fill_kernel_self(
    inp, expand_mask, value, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    rows_offset = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offset = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = rows_offset < M and cols_offset < N

    offsets = rows_offset * N + cols_offset
    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_val = tl.full((BLOCK_M, BLOCK_N), value, dtype=inp.dtype)
    tl.store(inp + offsets, cur_val, fill_mask and mask)


def masked_fill(inp, mask, value):
    logging.debug("GEMS MASKED FILL")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        # inp is a single-value
        return (
            torch.tensor(value, dtype=inp.dtype, device=inp.device)
            if mask.item()
            else inp.clone()
        )

    inp = inp.contiguous()
    mask = mask.contiguous()
    expand_mask = mask.expand(inp.shape)
    out = torch.empty_like(inp, dtype=inp.dtype, device=inp.device)

    index = (inp.ndim - 1) if inp.ndim > 0 else 0
    N = inp.size(index)
    if N == 0:
        return out
    M = inp.numel() // N
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    masked_fill_kernel[grid](inp, expand_mask.to(torch.int), value, out, M, N)
    return out


def masked_fill_(inp, mask, value):
    logging.debug("GEMS MASKED FILL")
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        # Value can be a tensor or a scalar
        value = value.item()
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        # inp is a single-value
        if mask.item():
            inp[()] = value
        return inp

    inp = inp.contiguous()
    mask = mask.contiguous()
    expand_mask = mask.expand(inp.shape)

    index = (inp.ndim - 1) if inp.ndim > 0 else 0
    N = inp.size(index)
    if N == 0:
        return inp
    M = inp.numel() // N
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    masked_fill_kernel_self[grid](inp, expand_mask.to(torch.int), value, M, N)
    return inp
