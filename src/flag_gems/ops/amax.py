import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import dim_compress, libentry, cfggen_reduce_op, TOTAL_CORE_NUM
from ..utils.shape_utils import can_use_int32_index


@libentry()
@triton.autotune(configs=cfggen_reduce_op(), key=["M"])
@triton.jit
def amax_kernel_1(
    inp,
    out,
    FILL_VALUE,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    _tmp = tl.full([BLOCK_SIZE], value=FILL_VALUE, dtype=inp.dtype.element_ty)
    block_start = block_start.to(tl.int64)
    for off in range(block_start, M, step):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=FILL_VALUE)
        _tmp = tl.where((_tmp < inp_val), inp_val, _tmp)

    amax_val = tl.max(_tmp)
    tl.atomic_max(out, amax_val)


def cfggen_opt():
    tile_num_n = [1, 2, 4, 8, 16, 48]
    num_stage = [1, 3]
    configs = [
        triton.Config({"TILE_NUM_N": n}, num_warps=1, num_stages=s) for n in tile_num_n for s in num_stage
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen_opt(), key=["N"])
@triton.jit
def amax_kernel_opt(
    inp,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    TILE_NUM_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    # Map the program id to the row of inp it should compute.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if INT64_INDEX:
        pid_m = pid_m.to(tl.int64)
        pid_n = pid_n.to(tl.int64)

    num_jobs = tl.num_programs(0)
    rows_per_job = (M + num_jobs -1) // num_jobs
    remain_rows = rows_per_job * num_jobs - M
    row_begin = pid_m * rows_per_job
    row_end = row_begin + rows_per_job
    if pid_m == (num_jobs - 1):
        row_end = row_begin + rows_per_job - remain_rows

    BLOCK_N: tl.constexpr = (N + TILE_NUM_N - 1) // TILE_NUM_N

    for row_idx in range(row_begin, row_end):
        offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        inp_ptrs = inp + row_idx * N + offset_n
        mask = offset_n < N
        inps = tl.load(inp_ptrs, mask, other=-float("inf"))
        max_val = tl.max(inps)
        new_out = out + row_idx
        tl.atomic_max(new_out, max_val)


def cfggen():
    block_m = [1, 2, 4, 8]
    block_n = [1024, 2048, 4096]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=1, num_stages=3) for m in block_m for n in block_n
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    # Map the program id to the row of inp it should compute.
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)

    num_jobs = tl.num_programs(axis=0)
    start_m = pid * BLOCK_M
    step = num_jobs * BLOCK_M
    for off_m in range(start_m, M, step):
        rows = off_m + tl.arange(0, BLOCK_M)[:, None]
        new_inp = inp + rows * N
        new_out = out + rows
        row_mask = rows < M

        _all = tl.full([BLOCK_M, BLOCK_N], value=-float("inf"), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(new_inp + cols, mask, other=-float("inf"))
            _all = tl.maximum(a, _all)

        all = tl.max(_all, axis=1)[:, None]
        tl.store(new_out, all, row_mask)


def amax(inp, dim=None, keepdim=False):
    logging.debug("GEMS AMAX")
    if dim is None or len(dim) == 0:
        M = inp.numel()
        grid = lambda meta: (min(triton.cdiv(M, meta['BLOCK_SIZE']), TOTAL_CORE_NUM), )
        dtype = inp.dtype
        use_int64_index = not can_use_int32_index(inp)
        if not keepdim:
            out = torch.full([], float("-inf"), dtype=torch.float32, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.full(shape, float("-inf"), dtype=torch.float32, device=inp.device)
        with torch.cuda.device(inp.device):
            fill_value = torch.finfo(inp.dtype).min
            amax_kernel_1[grid](inp, out, fill_value, M, INT64_INDEX=use_int64_index)
        return out.to(dtype)
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        use_int64_index = not can_use_int32_index(inp)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        with torch.cuda.device(inp.device):
            if N > 1048576:
                out = torch.empty(shape, dtype=dtype, device=inp.device)
                grid = lambda meta: (min(triton.cdiv(M, meta['BLOCK_M']), TOTAL_CORE_NUM), )
                amax_kernel[grid](inp, out, M, N, INT64_INDEX=use_int64_index)
            else :
                out = torch.full(shape, -float("inf"), dtype=torch.float32, device=inp.device)
                grid = lambda meta: (triton.cdiv(TOTAL_CORE_NUM, meta["TILE_NUM_N"]), meta["TILE_NUM_N"])
                amax_kernel_opt[grid](inp, out, M, N, INT64_INDEX=use_int64_index)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out.to(dtype)
