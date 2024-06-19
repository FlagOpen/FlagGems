import logging

import torch
import triton
import triton.language as tl

from ..utils import dim_compress, libentry


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def select_kernel(inp, out, M, N, index, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    col_offset = index

    inp = inp + rows_offsets * N + col_offset
    out = out + rows_offsets

    selected = tl.load(inp, mask=rows_mask, other=0.0)
    tl.store(out, selected, rows_mask)


def select(inp, dim=None, index=None):
    logging.debug("GEMS SELECT")
    assert dim is not None and dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert (
        index is not None and index >= -inp.size(dim) and index < inp.size(dim)
    ), "Invalid index"
    dim = dim % inp.ndim
    index = index % inp.size(dim)
    dtype = inp.dtype
    inp_shape = list(inp.shape)

    inp = dim_compress(inp, dim)
    N = inp_shape[dim]
    M = inp.numel() // N
    out_shape = inp_shape
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=dtype, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    select_kernel[grid](inp, out, M, N, index)
    return out.squeeze(dim=dim)


inp = torch.arange(1, 26).reshape(5, 5)
print(select(inp))
