import logging

import torch
import triton
import triton.language as tl

from ..utils import dim_compress, libentry


def cfggen():
    block_m = [1, 2, 4, 8]
    block_n = [1024, 2048]
    warps = [4, 8, 16]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w)
        for m in block_m
        for n in block_n
        for w in warps
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
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Var = Var + pid
    Mean = Mean + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _count = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)

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
    tl.store(Mean, mean, row_mask)
    tl.store(Var, var, row_mask)


@libentry()
@triton.jit
def var_mean_kernel_1(
    X,
    Acc,
    Average,
    Count,
    N,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    X = X + offset
    Acc = Acc + pid
    Average = Average + pid
    Count = Count + pid
    mask = offset < N

    x = tl.load(X, mask, other=0.0).to(tl.float32)

    count = tl.sum(mask.to(tl.float32))
    average = tl.sum(x) / count
    acc = tl.sum(x * x) - count * average * average

    tl.store(Average, average)
    tl.store(Acc, acc)
    tl.store(Count, count)


@libentry()
@triton.heuristics(
    values={"BLOCK_N": lambda args: triton.next_power_of_2(args["BLOCK_NUM"])},
)
@triton.jit(do_not_specialize=["correction"])
def var_mean_kernel_2(
    Acc,
    Average,
    Count,
    Var,
    Mean,
    N,
    correction,
    BLOCK_NUM,
    BLOCK_N: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_N)
    mask = offset < BLOCK_NUM
    Acc = Acc + offset
    Average = Average + offset
    Count = Count + offset
    acc = tl.load(Acc, mask, other=0.0).to(tl.float32)
    average = tl.load(Average, mask, other=0.0).to(tl.float32)
    count = tl.load(Count, mask, other=0.0).to(tl.float32)

    mean, _, nvar = tl.reduce((average, count, acc), axis=0, combine_fn=welford_func)

    var = nvar / (N - correction)
    tl.store(Mean, mean)
    tl.store(Var, var)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    logging.debug("GEMS VAR MEAN")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)
        BLOCK_N = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N)
        acc = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        average = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        count = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)

        var_mean_kernel_1[(BLOCK_NUM,)](x, acc, average, count, N, BLOCK_N=BLOCK_N)
        var_mean_kernel_2[(1,)](
            acc, average, count, var, mean, N, correction, BLOCK_NUM
        )
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

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        var_mean_welford_kernel[grid](x, var, mean, M, N, correction)

    if not keepdim:
        var = var.squeeze(dim=dim)
        mean = mean.squeeze(dim=dim)
    return var, mean
