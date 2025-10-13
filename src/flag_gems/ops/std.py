import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

from .sqrt import sqrt

logger = logging.getLogger(__name__)


@triton.jit
def welford_combine_fn(mean_x, count_x, M_x, mean_y, count_y, M_y):
    if count_x == 0:
        return mean_y, count_y, M_y
    if count_y == 0:
        return mean_x, count_x, M_x
    count = count_x + count_y
    delta = mean_y - mean_x
    mean = mean_x + delta * (count_y / count)
    M = M_x + M_y + delta * delta * (count_x * count_y / count)
    return mean, count, M


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("var_mean"), key=["M", "N"])
@triton.jit(do_not_specialize=["correction"])
def _var_mean_dim_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = pid < M

    X_ptrs = X + pid[:, None] * N
    Var_ptrs = Var + pid
    Mean_ptrs = Mean + pid

    sum_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    sum_sq_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols[None, :] < N)

        x = tl.load(X_ptrs + cols[None, :], mask=mask, other=0.0).to(tl.float32)

        sum_acc += tl.sum(x, axis=1)
        sum_sq_acc += tl.sum(x * x, axis=1)

    mean = sum_acc / N
    var_biased = (sum_sq_acc / N) - (mean * mean)

    divisor = N - correction
    var = var_biased * (N / tl.where(divisor > 0, divisor, 1.0))
    var = tl.where(divisor > 0, var, 0.0)

    tl.store(Mean_ptrs, mean, row_mask)
    tl.store(Var_ptrs, var, row_mask)


@libentry()
@triton.jit
def _var_mean_kernel_1(X, Acc, Average, Count, N, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0)
    offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offset < N
    x = tl.load(X + offset, mask, other=0.0).to(tl.float32)
    count = tl.sum(mask.to(tl.float32))
    if count == 0:
        tl.store(Average + pid, 0.0)
        tl.store(Acc + pid, 0.0)
        tl.store(Count + pid, 0.0)
        return
    mean = tl.sum(x) / count
    x_minus_mean = x - mean
    acc = tl.sum(x_minus_mean * x_minus_mean * mask.to(tl.float32))
    tl.store(Average + pid, mean)
    tl.store(Acc + pid, acc)
    tl.store(Count + pid, count)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("var_mean"))
@triton.jit(do_not_specialize=["correction"])
def _var_mean_kernel_2(
    Acc,
    Average,
    Count,
    Var,
    Mean,
    N,
    correction: tl.constexpr,
    BLOCK_NUM,
    BLOCK_N: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_N)
    mask = offset < BLOCK_NUM
    acc_vals = tl.load(Acc + offset, mask, other=0.0).to(tl.float32)
    avg_vals = tl.load(Average + offset, mask, other=0.0).to(tl.float32)
    cnt_vals = tl.load(Count + offset, mask, other=0.0).to(tl.float32)
    mean, _, m2 = tl.reduce(
        (avg_vals, cnt_vals, acc_vals), axis=0, combine_fn=welford_combine_fn
    )
    divisor = N - correction
    var = m2 / tl.where(divisor > 0, divisor, 1.0)
    var = tl.where(divisor > 0, var, 0.0)
    tl.store(Mean, mean)
    tl.store(Var, var)


def _var_mean_common(
    x,
    dim=None,
    unbiased: bool = True,
    keepdim: bool = False,
):
    correction = 1.0 if unbiased else 0.0

    if dim is None:
        dim_list = tuple(range(x.ndim))
        out_shape = [1] * x.ndim
        N = x.numel()
        if N == 0:
            nan_tensor = torch.full(
                out_shape, float("nan"), dtype=x.dtype, device=x.device
            )
            return nan_tensor, nan_tensor
        var = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        with torch_device_fn.device(x.device):
            BLOCK_N_1 = 1024
            BLOCK_NUM = triton.cdiv(N, BLOCK_N_1)
            mid_dtype = torch.float32
            acc = torch.empty(BLOCK_NUM, dtype=mid_dtype, device=x.device)
            average = torch.empty(BLOCK_NUM, dtype=mid_dtype, device=x.device)
            count = torch.empty(BLOCK_NUM, dtype=mid_dtype, device=x.device)
            _var_mean_kernel_1[(BLOCK_NUM,)](
                x, acc, average, count, N, BLOCK_N=BLOCK_N_1
            )
            _var_mean_kernel_2[(1,)](
                acc, average, count, var, mean, N, correction, BLOCK_NUM
            )

    else:
        shape = list(x.shape)
        if isinstance(dim, int):
            dim_list = [dim]
        else:
            dim_list = list(dim)
        dim_list = tuple(d % x.ndim for d in dim_list)
        out_shape = list(shape)
        N_reduce = 1
        for i in dim_list:
            N_reduce *= shape[i]
            out_shape[i] = 1
        if N_reduce == 0:
            final_shape = (
                out_shape
                if keepdim
                else torch.squeeze(torch.empty(out_shape), dim=dim_list).shape
            )
            nan_tensor = torch.full(
                final_shape, float("nan"), dtype=x.dtype, device=x.device
            )
            return nan_tensor, nan_tensor
        x_compressed = dim_compress(x, dim_list)
        M_rest = x_compressed.numel() // N_reduce
        var = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M_rest, META["BLOCK_M"]),)
        with torch_device_fn.device(x.device):
            _var_mean_dim_kernel[grid](
                x_compressed, var, mean, M_rest, N_reduce, correction
            )

    if not keepdim:
        var = var.squeeze(dim=dim_list)
        mean = mean.squeeze(dim=dim_list)

    return var, mean


def std(x, dim=None, *, unbiased=True, keepdim=False):
    logger.debug("GEMS STD")
    variance, _ = _var_mean_common(x, dim, unbiased, keepdim)
    return sqrt(variance)
