import logging

import torch
import triton
import triton.language as tl
import logging
from ..utils import dim_compress, libentry, cfggen_reduce_op, TOTAL_CORE_NUM


def cfggen():
    block_m = [1, 2, 4, 8]
    block_n = [1024, 2048]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=1)
        for m in block_m
        for n in block_n
    ]
    return configs

@triton.jit
def welford_func(mean_x, count_x, M_x, mean_y, count_y, M_y):
    count = count_x + count_y
    _count = tl.maximum(count, 1)
    mc_x = mean_x * count_x
    mc_y = mean_y * count_y
    mean = (mc_x + mc_y) / _count
    M = M_x + mc_x * mean_x + M_y + mc_y * mean_y - count * mean * mean
    return mean, count, M


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit(do_not_specialize=["correction"])
def var_mean_welford_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Var_ptr = Var + pid
        Mean_ptr = Mean + pid
        row_mask = pid < M

        _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        _acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        _count = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            x = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)

            count = _count + mask
            cnt = tl.maximum(count, 1)
            cur_mean = (_mean * _count + x) / cnt
            _acc += (x - cur_mean) * (x - _mean) * mask
            _mean = cur_mean
            _count = count

        mean, _, acc = tl.reduce((_mean, _count, _acc), axis=1, combine_fn=welford_func)
        var = acc / (N - correction)
        mean = mean[:, None]
        var = var[:, None]
        # Write mean / var
        tl.store(Mean_ptr, mean, row_mask)
        tl.store(Var_ptr, var, row_mask)


@libentry()
@triton.autotune(configs=cfggen_reduce_op(), key=["M"])
@triton.jit
def var_mean_kernel_1(
    X,
    Acc,
    Average,
    Count,
    M,
    BLOCK_SIZE: tl.constexpr,
):

    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE

    count = 0.0
    _tmp1 = tl.zeros([BLOCK_SIZE], tl.float32)
    _tmp2 = tl.zeros([BLOCK_SIZE], tl.float32)
    for block_start_offset in range(block_start, M, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        _count = tl.sum(mask.to(tl.float32))
        count = count + _count
        _tmp1 = _tmp1 + x
        _tmp2 = _tmp2 + x * x

    count = tl.maximum(count, 1)
    average = tl.sum(_tmp1) / count
    acc = tl.sum(_tmp2) - count * average * average

    Acc = Acc + pid
    Average = Average + pid
    Count = Count + pid

    tl.store(Average, average)
    tl.store(Acc, acc)
    tl.store(Count, count)


@libentry()
@triton.jit(do_not_specialize=["correction"])
def var_mean_kernel_2(
    Acc,
    Average,
    Count,
    Var,
    Mean,
    M,
    correction,
    BLOCK_NUM: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_NUM)
    Acc = Acc + offset
    Average = Average + offset
    Count = Count + offset
    acc = tl.load(Acc)
    average = tl.load(Average)
    count = tl.load(Count)
    mean, _, nvar = tl.reduce((average, count, acc), axis=0, combine_fn=welford_func)

    var = nvar / (M - correction)
    tl.store(Mean, mean)
    tl.store(Var, var)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    logging.debug("GEMS VAR MEAN")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        M = x.numel()

        grid = lambda meta: (min(triton.cdiv(M, meta['BLOCK_SIZE']), TOTAL_CORE_NUM), )
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)
        acc = torch.empty([TOTAL_CORE_NUM], dtype=torch.float, device=x.device)
        average = torch.empty([TOTAL_CORE_NUM], dtype=torch.float, device=x.device)
        count = torch.empty([TOTAL_CORE_NUM], dtype=torch.float, device=x.device)
        with torch.cuda.device(x.device):
            var_mean_kernel_1[(TOTAL_CORE_NUM,)](x, acc, average, count, M)
            var_mean_kernel_2[(1,)](
                acc, average, count, var, mean, M, correction, BLOCK_NUM=TOTAL_CORE_NUM)
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)

        grid = lambda META: (min(triton.cdiv(M, META["BLOCK_M"]), TOTAL_CORE_NUM),)
        with torch.cuda.device(x.device):
            var_mean_welford_kernel[grid](x, var, mean, M, N, correction)

    if not keepdim:
        var = var.squeeze(dim=dim)
        mean = mean.squeeze(dim=dim)
    return var, mean
