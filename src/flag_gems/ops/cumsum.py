import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry

MAX_C_MLU_CUMSUM = 16384

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
        "K",
    ],
)
@triton.heuristics(
    values={"BLOCK_N": lambda args: triton.next_power_of_2(args["N"])},
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



@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_CUMSUM//4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_CUMSUM//4}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": MAX_C_MLU_CUMSUM//4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": MAX_C_MLU_CUMSUM//4}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": MAX_C_MLU_CUMSUM//4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": MAX_C_MLU_CUMSUM//4}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_CUMSUM//2}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_CUMSUM//2}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": MAX_C_MLU_CUMSUM//2}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": MAX_C_MLU_CUMSUM//2}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": MAX_C_MLU_CUMSUM//2}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": MAX_C_MLU_CUMSUM//2}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_CUMSUM}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_CUMSUM}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": MAX_C_MLU_CUMSUM}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": MAX_C_MLU_CUMSUM}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": MAX_C_MLU_CUMSUM}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": MAX_C_MLU_CUMSUM}, num_warps=8, num_stages=5),
    ],
    key=[
        "M",
        "N",
        "K",
    ],
)
@triton.jit
def cumsum_kernel_split(
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

    kep = tl.full([BLOCK_M], float(0), tl.float32)
    for col_offset in range(0, N, BLOCK_N):
        n_offset = col_offset + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * \
            K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=float(0)).to(tl.float32)

        raw_res = tl.cumsum(inp_vals, axis=1)
        result = raw_res + kep[:, None]
        kep = result[:, BLOCK_N-1]

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
    out = torch.empty_like(inp, dtype=dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    if N > MAX_C_MLU_CUMSUM:
        logging.debug("GEMS CUMSUM USE SPLITC FOR N = %d" % (N))
        cumsum_kernel_split[grid](inp, out, M, N, K)
    else:
        cumsum_kernel[grid](inp, out, M, N, K)
    return out
