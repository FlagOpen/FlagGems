import logging
import math

import torch
import triton
import triton.language as tl

from .. import runtime
from ..utils import libentry
from ..utils import triton_lang_extension as tle
from ..utils.type_utils import get_accumulator_dtype


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.autotune(
    configs=runtime.get_triton_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # using 1d tile makes code clean
    # Map the program id to the row of X and Y it should compute.
    pid = tle.program_id(0)

    n_offsets = tl.arange(0, TILE_N)
    mask = n_offsets < N

    x = tl.load(in_ptr + pid * N + n_offsets, mask, other=0.0).to(tl.float32)
    m = tl.sum(x) / N
    d = x - m  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    w = tl.load(weight_ptr + n_offsets, mask=mask)
    b = tl.load(bias_ptr + n_offsets, mask=mask)
    out = (x - m) * rstd * w + b

    tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_triton_config("layer_norm_persistent"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_persistent_kernel_multiline(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tle.program_id(0)
    m_offsets = pid * TILE_M + tl.arange(0, TILE_M)
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, TILE_N)[None, :]
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask

    x = tl.load(in_ptr + m_offsets[:, None] * N + n_offsets, mask, other=0.0).to(
        tl.float32
    )
    m = tl.sum(x, axis=1) / N
    d = x - m[:, None]  # deviation
    s = tl.where(mask, d * d, 0)
    sum_square = tl.sum(s, axis=1)  # sum of square of deviation
    var = sum_square / N
    rstd = tl.math.rsqrt(var + eps)

    tl.store(out_mean_ptr + m_offsets, m, mask=m_mask)
    tl.store(out_rstd_ptr + m_offsets, rstd, mask=m_mask)

    w = tl.load(weight_ptr + n_offsets, mask=n_mask)
    b = tl.load(bias_ptr + n_offsets, mask=n_mask)
    out = (x - m[:, None]) * rstd[:, None] * w + b

    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_triton_config("layer_norm_loop"),
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def layer_norm_loop_kernel(
    in_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    out_mean_ptr,  # pointer to the mean
    out_rstd_ptr,  # pointer to the 1/std
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tle.program_id(0)

    # Compute mean
    m = tl.zeros((TILE_N,), dtype=tl.float32)  # mean
    s = tl.zeros((TILE_N,), dtype=tl.float32)  # sum((x - m)^2)
    cnt = tl.zeros((TILE_N,), dtype=tl.int32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        new_m = m + (x - m) / (step + 1)
        new_s = s + (x - new_m) * (x - m)
        cnt += 1
        m = new_m
        s = new_s

    # the last step
    for step in range(num_steps - 1, num_steps, 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask).to(tl.float32)
        new_m = tl.where(mask, m + (x - m) / (step + 1), m)
        new_s = tl.where(mask, s + (x - new_m) * (x - m), s)
        cnt += mask.to(tl.int32)
        m = new_m
        s = new_s

    final_m = tl.sum(m * cnt) / N
    var = tl.sum(s + cnt * (m - final_m) * (m - final_m)) / N
    rstd = tl.math.rsqrt(var + eps)
    m = final_m
    # Write mean / rstd
    tl.store(out_mean_ptr + pid, m)
    tl.store(out_rstd_ptr + pid, rstd)

    # reverse the order of the second sweep
    # Normalize and apply linear transformation
    prev_multiple = prev_multiple_of(N, TILE_N)
    # the first step, masking is needed
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w = tl.load(weight_ptr + n_offsets, mask=mask)
        b = tl.load(bias_ptr + n_offsets, mask=mask)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets, eviction_policy="evict_first").to(
            tl.float32
        )
        w = tl.load(weight_ptr + n_offsets)
        b = tl.load(bias_ptr + n_offsets)
        out = w * (x - m) * rstd + b
        tl.store(out_ptr + pid * N + n_offsets, out)


@libentry()
@triton.autotune(
    configs=runtime.get_triton_config("layer_norm_backward"),
    key=["M", "N"],
)
@triton.jit
def layer_norm_backward_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tle.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M
    dY += pid * N
    X += pid * N
    dX += pid * N
    Mean += pid
    Rstd += pid

    mean = tl.load(Mean).to(tl.float32)
    rstd = tl.load(Rstd).to(tl.float32)

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2, axis=1)[:, None]
    dx_3 = tl.sum(dx_part3, axis=1)[:, None]

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask and col_mask
        dy = tl.load(dY + cols[None, :], mask).to(tl.float32)
        x = tl.load(X + cols[None, :], mask).to(tl.float32)
        w = tl.load(W + cols, mask=cols < N).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX + cols, dx, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_triton_config("weight_bias_backward"),
    key=["N"],
)
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tle.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)[None, :]
    col_mask = pid < N
    dY += pid
    X += pid
    dW += pid
    dB += pid
    accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)
        row_mask = rows[:, None] < M
        mask = row_mask and col_mask
        dy = tl.load(dY + rows[:, None] * N, mask).to(tl.float32)
        x = tl.load(X + rows[:, None] * N, mask).to(tl.float32)
        mean = tl.load(Mean + rows, mask=rows < M)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=rows < M)[:, None].to(tl.float32)
        x = tl.where(col_mask, x - mean, 0.0)
        x_hat = x * rstd
        accW += dy * x_hat
        accB += dy
    dw = tl.sum(accW, axis=0)
    db = tl.sum(accB, axis=0)
    tl.store(dW, dw[None, :], mask=col_mask)
    tl.store(dB, db[None, :], mask=col_mask)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps=1e-5, cudnn_enable=True):
        logging.debug("GEMS LAYERNORM FORWARD")
        # dim = x.ndim - len(normalized_shape)
        # M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)
        M = x.numel() // N

        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        # NOTE: when the input is half-precision(either float16 or bfloat16)
        # these statistical data saved for backward is in single precision
        acc_type = get_accumulator_dtype(x.dtype)
        mean = torch.empty(M, dtype=acc_type, device=x.device)
        rstd = torch.empty(M, dtype=acc_type, device=x.device)

        with torch.cuda.device(x.device):
            if N <= 128:
                TILE_N = triton.next_power_of_2(N)
                TILE_M = triton.cdiv(1024, TILE_N)
                grid = (triton.cdiv(M, TILE_M), 1, 1)
                layer_norm_persistent_kernel_multiline[grid](
                    x,
                    y,
                    weight,
                    bias,
                    mean,
                    rstd,
                    M,
                    N,
                    eps,
                    TILE_M,
                    TILE_N,
                )
            elif N <= 4096:
                TILE_N = triton.next_power_of_2(N)
                grid = (M, 1, 1)
                layer_norm_persistent_kernel[grid](
                    x,
                    y,
                    weight,
                    bias,
                    mean,
                    rstd,
                    M,
                    N,
                    eps,
                    TILE_N,
                )
            else:
                grid = (M, 1, 1)
                layer_norm_loop_kernel[grid](
                    x,
                    y,
                    weight,
                    bias,
                    mean,
                    rstd,
                    M,
                    N,
                    eps,
                )
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.M = M
        ctx.N = N
        return y, mean, rstd

    @staticmethod
    def backward(ctx, out_grad, mean_grad, rstd_grad):
        logging.debug("GEMS LAYERNORM BACKWARD")
        out_grad = out_grad.contiguous()
        (x, weight, mean, rstd) = ctx.saved_tensors
        M = ctx.M
        N = ctx.N

        with torch.cuda.device(x.device):
            in_grad = torch.empty_like(x)
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]), 1, 1)
            layer_norm_backward_kernel[grid](
                out_grad, x, weight, mean, rstd, in_grad, M, N
            )

            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
            weight_grad = torch.empty_like(weight)
            bias_grad = torch.empty_like(weight)
            weight_bias_backward_kernel[grid](
                out_grad, x, mean, rstd, weight_grad, bias_grad, M, N
            )
        return in_grad, None, weight_grad, bias_grad, None, None


def layer_norm(x, normalized_shape, weight, bias, eps=1e-5, cudnn_enable=True):
    return LayerNorm.apply(x, normalized_shape, weight, bias, eps, cudnn_enable)
