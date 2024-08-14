import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@triton.autotune(
    configs=[
        # triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=4),
        # triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=5),
        # triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=4),
        # triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=5),
        # triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 512}, num_warps=8, num_stages=5),
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
def cumsum_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    for xoffset in range(pid_m * BLOCK_M, pid_m * BLOCK_M + BLOCK_M, 1):
        row_mask = xoffset < M
        sum_base = 0.0
        for yoffset in range(0, BLOCK_N, 1):
            col_mask = yoffset < N
            mask = row_mask and col_mask
            inp_ptrs = inp + xoffset * N + yoffset
            inp_vals = tl.load(inp_ptrs, mask).to(tl.float32)
            sum_base = sum_base + inp_vals
            out_ptrs = out + xoffset * N + yoffset
            tl.store(out_ptrs, sum_base, mask)


def cumsum(inp, dim=1, *, dtype=None):
    logging.debug("GEMS CUMSUM")
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
        cumsum_kernel[grid](inp, out, M, N, K)
    return out
