import logging
import math

import torch
import triton
import triton.language as tl
import logging
from ..utils import dim_compress, libentry, cfggen_reduce_op, TOTAL_CORE_NUM

try:
    from triton.language.extra.mlu.libdevice import pow
except ImportError:
    try:
        from triton.language.math import pow
    except ImportError:
        from triton.language.libdevice import pow


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=1) for m in block_m
    ]
    return configs

@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def l2_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    if task_num % num_prog != 0:
        iter_num = iter_num + 1
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _sum += a * a
        sum = tl.sum(_sum, axis=1)

        out = tl.sqrt(sum)[:, None]
        tl.store(Out_ptr, out, row_mask)

@libentry()
@triton.autotune(configs=cfggen_reduce_op(), key=["M"])
@triton.jit
def l2_norm_kernel_1(
    X,
    Mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE

    _tmp = tl.zeros([BLOCK_SIZE], tl.float32)
    for block_start_offset in range(block_start, M, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        _tmp = _tmp + x * x

    mid = tl.sum(_tmp)
    Mid = Mid + pid
    tl.store(Mid, mid)

@libentry()
@triton.jit
def l2_norm_kernel_2(
    Mid,
    Out,
    BLOCK_NUM: tl.constexpr
):
    offset = tl.arange(0, BLOCK_NUM)
    Mid = Mid + offset
    mid = tl.load(Mid)
    out = tl.sqrt(tl.sum(mid))
    tl.store(Out, out)


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def max_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    if task_num % num_prog != 0:
        iter_num = iter_num + 1
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _max = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _max = tl.maximum(tl.abs(a), _max)

        max = tl.max(_max, axis=1)
        out = max[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.autotune(configs=cfggen_reduce_op(), key=["M"])
@triton.jit
def max_norm_kernel_1(
    X,
    Out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE

    _tmp = tl.zeros([BLOCK_SIZE], tl.float32)
    for block_start_offset in range(block_start, M, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        _x = tl.abs(x)
        _tmp = tl.where(_tmp > _x, _tmp, _x)

    mid = tl.max(_tmp)
    tl.atomic_max(Out, mid.to(tl.float32))


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def min_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):

    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    if task_num % num_prog != 0:
        iter_num = iter_num + 1
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _min = tl.full([BLOCK_M, BLOCK_N], value=float("inf"), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=float("inf")).to(tl.float32)
            _min = tl.minimum(tl.abs(a), _min)

        min = tl.min(_min, axis=1)
        out = min[:, None]
        tl.store(Out_ptr, out, row_mask)

@libentry()
@triton.autotune(configs=cfggen_reduce_op(), key=["M"])
@triton.jit
def min_norm_kernel_1(
    X,
    Out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE

    _tmp = tl.zeros([BLOCK_SIZE], tl.float32)
    for block_start_offset in range(block_start, M, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=float("inf")).to(tl.float32)
        _x = tl.abs(x)
        _tmp = tl.where(_tmp < _x, _tmp, _x)

    mid = tl.min(_tmp)
    tl.atomic_min(Out, mid.to(tl.float32))


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def l0_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    if task_num % num_prog != 0:
        iter_num = iter_num + 1
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=0).to(tl.float32)
            _sum += tl.where(a != 0, 1, 0)
        sum = tl.sum(_sum, axis=1)
        out = sum[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.autotune(configs=cfggen_reduce_op(), key=["M"], reset_to_zero=['Out'])
@triton.jit
def l0_norm_kernel_1(
    X,
    Out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE

    _tmp = tl.zeros([BLOCK_SIZE], tl.float32)
    for block_start_offset in range(block_start, M, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        _tmp = _tmp + (x != 0).to(tl.float32)

    mid = tl.sum(_tmp)
    tl.atomic_add(Out, mid.to(tl.float32))


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit(do_not_specialize=["ord"])
def v_norm_kernel(X, Out, M, N, ord, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)

    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Out_ptr = Out + pid
        row_mask = pid < M

        _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _sum += tl.extra.mlu.libdevice.pow(tl.abs(a), ord)
        sum = tl.sum(_sum, axis=1)
        out = tl.extra.mlu.libdevice.pow(sum, 1 / ord)[:, None]
        tl.store(Out_ptr, out, row_mask)


@libentry()
@triton.autotune(configs=cfggen_reduce_op(), key=["M"])
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_1(
    X,
    Mid,
    M,
    ord,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE

    _tmp = tl.zeros([BLOCK_SIZE], tl.float32)
    for block_start_offset in range(block_start, M, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        _tmp = _tmp + pow(tl.abs(x), ord)

    mid = tl.sum(_tmp)
    Mid = Mid + pid
    tl.store(Mid, mid)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def l1_norm_kernel_2(
    Mid,
    Out,
    ord,
    BLOCK_NUM: tl.constexpr
):
    offset = tl.arange(0, BLOCK_NUM)
    Mid = Mid + offset
    mid = tl.load(Mid)
    out = pow(tl.sum(mid), 1 / ord)
    tl.store(Out, out)


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    logging.debug("GEMS VECTOR NORM")
    if dtype is not None:
        dtype = torch.dtype(dtype)
    else:
        dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"vector_norm not implemented for {dtype}")

    with torch.mlu.device(x.device):
        if dim is None or len(dim) == x.ndim:
            dim = list(range(x.ndim))
            shape = [1] * x.ndim
            x = dim_compress(x, dim)
            M = x.numel()

            grid = lambda meta: (min(triton.cdiv(M, meta['BLOCK_SIZE']), TOTAL_CORE_NUM), )
            mid = torch.empty([TOTAL_CORE_NUM], dtype=torch.float, device=x.device)
            out = torch.zeros(shape, dtype=torch.float, device=x.device)
            if ord == 2:
                l2_norm_kernel_1[(TOTAL_CORE_NUM,)](x, mid, M)
                l2_norm_kernel_2[(1,)](mid, out, BLOCK_NUM=TOTAL_CORE_NUM)
            elif ord == float("inf"):
                max_norm_kernel_1[grid](x, out, M)
            elif ord == -float("inf"):
                min_norm_kernel_1[grid](x, out, M)
            elif ord == 0:
                l0_norm_kernel_1[grid](x, out, M)
            else:
                l1_norm_kernel_1[(TOTAL_CORE_NUM,)](x, mid, M, ord)
                l1_norm_kernel_2[(1,)](mid, out, ord, BLOCK_NUM=TOTAL_CORE_NUM)
            out = out.to(dtype)
        else:
            shape = list(x.shape)
            dim = [d % x.ndim for d in dim]
            x = dim_compress(x, dim)
            N = 1
            for i in dim:
                N *= shape[i]
                shape[i] = 1
            M = x.numel() // N
            out = torch.empty(shape, dtype=dtype, device=x.device)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
            if ord == 2:
                l2_norm_kernel[grid](x, out, M, N)
            elif ord == float("inf"):
                max_norm_kernel[grid](x, out, M, N)
            elif ord == -float("inf"):
                min_norm_kernel[grid](x, out, M, N)
            elif ord == 0:
                l0_norm_kernel[grid](x, out, M, N)
            else:
                v_norm_kernel[grid](x, out, M, N, ord)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
