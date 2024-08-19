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
        triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=5),
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
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None, None] * N * K + n_offset[None, :, None] * K + pid_k
    mask = m_offset[:, None, None] < M and n_offset[None, :, None] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask).to(tl.float32)
    result = tl.cumsum(inp_vals, axis=1)
    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)


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


@triton.jit(do_not_specialize=["K"])
def fused_renorm_cumsum_kernel(inp, out, K, BLOCK: tl.constexpr):
    row_start = inp + tl.program_id(0) * K
    row_off = tl.arange(0, BLOCK)
    x = tl.load(row_start + row_off, mask=row_off < K)
    normed_x = x / tl.sum(x, 0)
    y = tl.cumsum(normed_x, 0)
    tl.store(out + row_off, y, mask=row_off < K)


def fused_renorm_cumsum(inp, dim=-1):
    logging.debug("GEMS RENORM_CUMSUM")
    assert inp.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
    dim = dim % inp.ndim
    assert dim == inp.ndim - 1, "Currently only supports the last dimension."
    inp = inp.contiguous()
    K = inp.size(dim)
    N = inp.numel()
    out = torch.empty_like(inp)
    with torch.cuda.device(inp.device.index):
        grid = lambda meta: (N // K,)
        BLOCK = triton.next_power_of_2(K)
        fused_renorm_cumsum_kernel[grid](inp, out, K, BLOCK=BLOCK)

    return out
