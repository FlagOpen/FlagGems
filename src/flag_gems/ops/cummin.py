import logging

import torch
import triton
import triton.language as tl
import triton.language.core as core

from ..utils import libentry


@triton.jit
def _min_combine(a, b):
    if a <= b:
        return a
    else:
        return b


@triton.jit
@core._add_scan_docstr("cummin")
def cummin_kernel_(input, axis=0):
    input = core._promote_reduction_input(input)
    return core.associative_scan(input, axis, _min_combine)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 32}, num_warps=8),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def cummin_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None, None] * N * K + n_offset[None, :, None] * K + pid_k
    mask = m_offset[:, None, None] < M and n_offset[None, :, None] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask).to(tl.float32)
    result = cummin_kernel_(inp_vals, axis=1)
    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)


def cummin(inp, dim=1, *, dtype=None):
    logging.debug("GEMS cummin")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64
    out = torch.empty_like(inp, dtype=dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.cuda.device(inp.device):
        cummin_kernel[grid](inp, out, M, N, K)
    return out
