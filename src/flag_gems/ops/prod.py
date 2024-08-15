import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry


@triton.jit
def reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset < M
    mid_value = 1.0
    for roffset in range(0, BLOCK_SIZE, 1):
        inp_ptrs = inp + offset + roffset
        inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
        mid_value = mid_value * inp_val
    mid_ptr = mid + pid
    tl.store(mid_ptr, mid_value)


@libentry()
@triton.jit
def prod_kernel_result(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    prod_val = 1.0
    for roffset in range(0, mid_size, 1):
        mask = roffset < mid_size
        mid_ptrs = mid + roffset
        mid_val = tl.load(mid_ptrs, mask=mask, other=1.0).to(tl.float32)
        prod_val = prod_val * mid_val
    tl.store(out, prod_val)


def prod(inp, *, dtype=None):
    logging.debug("GEMS PROD")
    if dtype is None:
        dtype = inp.dtype

    M = inp.numel()
    # block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    # mid_size = triton.cdiv(M, block_size)
    mid_size = 12  # CLUSTER_NUM
    block_size = triton.next_power_of_2(triton.cdiv(M, mid_size))
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device).to(torch.float32)
    out = torch.empty([], dtype=dtype, device=inp.device)
    final_mid_size = min(math.ceil(inp.numel() / block_size), min(mid_size, M))

    with torch.cuda.device(inp.device):
        prod_kernel_mid[(mid_size, 1, 1)](inp, mid, M, block_size)
        prod_kernel_result[(1, 1, 1)](mid, out, final_mid_size, block_mid)
    return out


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
def prod_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tl.program_id(0)

    for xoffset in range(pid_m * BLOCK_M, pid_m * BLOCK_M + BLOCK_M, 1):
        row_mask = xoffset < M
        prod_base = 1.0
        for yoffset in range(0, BLOCK_N, 1):
            col_mask = yoffset < N
            mask = row_mask and col_mask
            inp_ptrs = inp + xoffset * N + yoffset
            inp_vals = tl.load(inp_ptrs, mask).to(tl.float32)
            prod_base = prod_base * inp_vals
        out_ptrs = out + xoffset
        tl.store(out_ptrs, prod_base, row_mask)


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logging.debug("GEMS PROD DIM")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1

    if dtype is None:
        dtype = inp.dtype
    out = torch.empty(shape_list, dtype=dtype, device=inp.device)
    if not keepdim:
        out = torch.squeeze(out, dim)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.cuda.device(inp.device):
        prod_kernel[grid](inp, out, M, N, K)

    return out
