import logging

import torch
import triton
import triton.language as tl

from ..utils import broadcastable_to, libentry


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 128}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def masked_fill_kernel(
    inp, expand_mask, value, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    rows_offset = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offset < M

    for off in range(0, N, BLOCK_N):
        cols_offset = off + tl.arange(0, BLOCK_N)[None, :]
        cols_mask = cols_offset < N
        block_mask = rows_mask and cols_mask

        offsets = rows_offset * N + cols_offset
        fill_mask = tl.load(expand_mask + offsets, mask=block_mask, other=0).to(tl.int1)
        cur_inp = tl.load(inp + offsets, mask=(not fill_mask) and block_mask, other=0)
        tl.store(out + offsets, cur_inp, (not fill_mask) and block_mask)

        cur_val = tl.full((BLOCK_M, BLOCK_N), value, dtype=cur_inp.dtype)
        tl.store(out + offsets, cur_val, fill_mask and block_mask)


def masked_fill(inp, mask, value):
    logging.debug("GEMS MASKED FILL")
    assert (
        isinstance(value, float)
        or isinstance(value, int)
        or (torch.is_tensor(value) and value.ndim == 0)
    ), "masked_fill_ only supports a Number or a 0-dimensional value tensor"
    if torch.is_tensor(value):
        value = value.item()
    inp_shape = tuple(inp.shape)
    mask_shape = tuple(mask.shape)
    assert broadcastable_to(
        mask_shape, inp_shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    inp = inp.contiguous()
    mask = mask.contiguous()
    value = value.contiguous()
    expand_mask = mask.expand(inp.shape)
    out = torch.empty_like(inp, dtype=inp.dtype, device=inp.device)

    N = inp.size(inp.ndim - 1)
    M = inp.numel() // N
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    masked_fill_kernel[grid](inp, expand_mask.to(torch.int), value, out, M, N)
    return out
