import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from ..utils import libentry, cfggen_reduce_op, prune_reduce_config, TOTAL_CORE_NUM


@libentry()
@triton.jit
def min_kernel_float_once(
    inp,
    out,
    M: tl.constexpr,
):
    offset = tl.arange(0, M)
    inp_val = tl.load(inp + offset)
    min_val = tl.min(inp_val, 0, return_indices=True)
    tl.store(out, min_val)


@libentry()
@triton.autotune(
    configs=cfggen_reduce_op(),
    key=["M"],
    prune_configs_by = {'early_config_prune': prune_reduce_config}
)
@triton.heuristics(
    values={"ONE_TILE_PER_CTA": lambda args: args["M"] <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM}
)
@triton.jit
def min_kernel_float(
    inp,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    res = float("inf")
    if ONE_TILE_PER_CTA:
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask = mask, other=float("inf"))
        res = tl.min(inp_val, 0, return_indices=True)
    else:
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        _tmp = tl.full([BLOCK_SIZE], value=float("inf"), dtype=inp.dtype.element_ty)
        for off in range(block_start, M, step):
            offset = off + tl.arange(0, BLOCK_SIZE)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=float("inf"))
            _tmp = tl.where((inp_val < _tmp), inp_val, _tmp)
        res = tl.min(_tmp, 0, return_indices=True)
    tl.atomic_min(out, res)


@libentry()
@triton.autotune(
    configs=cfggen_reduce_op(),
    key=["M"],
    prune_configs_by = {'early_config_prune': prune_reduce_config}
)
@triton.heuristics(
    values={"ONE_TILE_PER_CTA": lambda args: args["M"] <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM}
)
@triton.jit
def min_kernel_int(
    inp,
    out,
    FILL_VALUE,
    M,
    BLOCK_SIZE: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    res = FILL_VALUE
    if ONE_TILE_PER_CTA:
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask = mask, other=FILL_VALUE)
        res = tl.min(inp_val)
    else:
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        block_start = block_start.to(tl.int64)
        _tmp = tl.full([BLOCK_SIZE], value=2**31 - 1, dtype=tl.int32)
        for off in range(block_start, M, step):
            offset = off + tl.arange(0, BLOCK_SIZE)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=FILL_VALUE)
            _tmp = tl.where((inp_val < _tmp), inp_val, _tmp)
        res = tl.min(_tmp)
    tl.atomic_min(out, res)


@libentry()
@triton.autotune(
    configs=cfggen_reduce_op(),
    key=["M"],
    prune_configs_by = {'early_config_prune': prune_reduce_config}
)
@triton.heuristics(
    values={"ONE_TILE_PER_CTA": lambda args: args["M"] <= args["BLOCK_SIZE"] * TOTAL_CORE_NUM}
)
@triton.jit
def min_kernel_int64_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # FILL_VALUE is the maximum value of a 64-bit integer, used as the initial value for calculations.
    FILL_VALUE = 2**63 - 1
    res = FILL_VALUE
    if ONE_TILE_PER_CTA:
        offset = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask = mask, other=FILL_VALUE)
        res = tl.min(inp_val)
    else:
        num_jobs = tl.num_programs(axis=0)
        step = num_jobs * BLOCK_SIZE
        block_start = block_start.to(tl.int64)
        _tmp = tl.full([BLOCK_SIZE], value=FILL_VALUE, dtype=tl.int64)
        for off in range(block_start, M, step):
            offset = off + tl.arange(0, BLOCK_SIZE)
            mask = offset < M
            inp_val = tl.load(inp + offset, mask=mask, other=FILL_VALUE)
            _tmp = tl.where((inp_val < _tmp), inp_val, _tmp)
        res = tl.min(_tmp)
    tl.store(mid + pid, res)

@libentry()
@triton.jit
def min_kernel_int64_2(
    mid,
    out,
    BLOCK_NUM: tl.constexpr
):
    offset = tl.arange(0, BLOCK_NUM)
    mid_val = tl.load(mid + offset)
    out_val = tl.min(mid_val)
    tl.store(out, out_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4}, num_warps=8, num_stages=4),
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
def min_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    offset_index = m_offset * K + pid_k
    # set mask
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=float("inf")).to(tl.float32)
    result_value, result_index = tl.min(inp_vals, axis=1, return_indices=True)

    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)


def min(inp):
    logging.debug("GEMS MIN")
    M = inp.numel()
    mid_size = TOTAL_CORE_NUM
    dtype = inp.dtype
    device = inp.device

    with torch.cuda.device(device):
        if torch.is_floating_point(inp):
            if M <= 65536:
                out = torch.empty([], dtype=dtype, device=device)
                min_kernel_float_once[(1, 1, 1)](inp, out, M)
            else:
                out = torch.full([], float("inf"), dtype=torch.float32, device=device)
                min_kernel_float[(mid_size, 1, 1)](inp, out, M)
        elif dtype == torch.int64:
            mid = torch.empty([mid_size], dtype=dtype, device=device)
            out = torch.empty([], dtype=dtype, device=device)
            # Because atomic op don't support i64, use two kernels.
            min_kernel_int64_1[(mid_size, 1, 1)](inp, mid, M, enable_soft_i64=True)
            min_kernel_int64_2[(1, 1, 1)](mid, out, BLOCK_NUM=mid_size, enable_soft_i64=True)
        else:
            fill_value = torch.iinfo(dtype).max
            out = torch.full([], 2**31 - 1, dtype=torch.int32, device=device)
            min_kernel_int[(mid_size, 1, 1)](inp, out, fill_value, M)
    return out.to(dtype)


def min_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS MIN DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.cuda.device(inp.device):
        min_kernel[grid](inp, out_value, out_index, M, N, K)
    Min_out = namedtuple("min", ["values", "indices"])
    out = Min_out(values=out_value, indices=out_index)
    return out
