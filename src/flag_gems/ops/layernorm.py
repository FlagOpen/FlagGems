import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry, TOTAL_CORE_NUM

MAX_C_MLU_LAYERNORM_BACKWARD = 8192

def cfggen():
    block_m = [i for i in range(1, 36, 4)]  #[1, 2, 4]
    block_n = [i for i in range(64, 193, 64)]
    warps = [1]
    num_stages = [1, 3]
    configs = [
        triton.Config({
            "BLOCK_ROW_SIZE": m,
            "BLOCK_COL_SIZE": n
        },
                      num_warps=w,
                      num_stages=s) for m in block_m for n in block_n
        for w in warps for s in num_stages
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit(do_not_specialize=["eps"])
def layer_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    M,
    N,
    eps,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid = tl.program_id(0)
    row = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = row < M
    X += row * N
    Y += row * N

    # Compute mean
    _mean = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    # Compute variance
    _var = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
        _var += a * a
    mean = tl.sum(_mean, axis=1) / N
    mean_bc = mean[:, None]

    var = tl.sum(_var, axis=1) / N - (mean * mean)
    var = var[:, None]
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean_bc, row_mask)
    tl.store(Rstd + row, rstd, row_mask)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        w = tl.load(W + cols, col_mask)
        b = tl.load(B + cols, col_mask)
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        x = tl.where(col_mask, x - mean_bc, 0.0)
        x_hat = x * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


def cfggen_bw():
    configs = [
        triton.Config({"BLOCK_ROW_SIZE": 1}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_ROW_SIZE": 1}, num_warps=1, num_stages=3),
        triton.Config({"BLOCK_ROW_SIZE": 4}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_ROW_SIZE": 4}, num_warps=1, num_stages=3),
        triton.Config({"BLOCK_ROW_SIZE": 16}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_ROW_SIZE": 32}, num_warps=1, num_stages=1),
    ]
    return configs

@libentry()
@triton.autotune(configs=cfggen_bw(), key=["M", "N"], reset_to_zero=["DW", "DB"])
@triton.jit
def layer_norm_backward_kernel(
        DX,  # pointer to the input gradient
        DY,  # pointer to the output gradient
        DW,  # pointer to the partial sum of weights gradient
        DB,  # pointer to the partial sum of biases gradient
        X,  # pointer to the input
        W,  # pointer to the weights
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        M,  # number of rows in X
        N,  # number of columns in X
        BLOCK_ROW_SIZE: tl.constexpr,
        BLOCK_COL_SIZE: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    pid = tl.program_id(0)

    row_start = pid * BLOCK_ROW_SIZE
    cols = tl.arange(0, BLOCK_COL_SIZE)
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_ROW_SIZE

    X += cols[None, :]
    DY += cols[None, :]
    W += cols[None, :]
    DX += cols[None, :]
    w = tl.load(W).to(tl.float32)

    partial_dw = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    partial_db = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for row in range(row_start, M, step):
        row_off = row + tl.arange(0, BLOCK_ROW_SIZE)
        mask = row_off[:, None] < M
        # Load data to SRAM
        off = row_off[:, None] * N
        x = tl.load(X + off, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + off, mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + row_off, mask = row_off < M)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + row_off, mask = row_off < M)[:, None].to(tl.float32)
        # Compute dx
        x_hat = (x - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=1)[:, None]
        c2 = tl.sum(wdy, axis=1)[:, None]
        dx = (wdy - (x_hat * c1 + c2) / N) * rstd
        # Accumulate partial sums for dw/db
        partial_dw += (dy * x_hat).to(tl.float32)
        partial_db += (dy).to(tl.float32)
        # Write dx
        tl.store(DX + off, dx.to(x.dtype), mask=mask)

    dw = tl.sum(partial_dw, axis=0)
    db = tl.sum(partial_db, axis=0)
    tl.atomic_add(DW + cols, dw)
    tl.atomic_add(DB + cols, db)


def cfggen_input_bw():
    block_m = [1, 4, 16, 32]
    block_n = [32, 256, 1024, 2048]
    num_stages = [1, 3]
    configs = [triton.Config({"BLOCK_ROW_SIZE": m, "BLOCK_COL_SIZE": n},
                             num_warps=1,
                             num_stages=s) for m in block_m for n in block_n for s in num_stages
    ]
    return configs

@libentry()
@triton.autotune(configs=cfggen_input_bw(), key=["M", "N"])
@triton.jit
def input_backward_kernel(
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
    pid = tl.program_id(0)

    row_start = pid * BLOCK_ROW_SIZE
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_ROW_SIZE

    for row in range(row_start, M, step):
        row_off = row + tl.arange(0, BLOCK_ROW_SIZE)
        mean = tl.load(Mean + row_off, mask = row_off < M, other = 0.0)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + row_off, mask = row_off < M, other = 0.0)[:, None].to(tl.float32)

        row_mask = row_off[:, None] < M
        off = row_off[:, None] * N
        new_dY = dY + off
        new_X = X + off
        new_DX = dX + off

        dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
        dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

        for off in range(0, N, BLOCK_COL_SIZE):
            cols = off + tl.arange(0, BLOCK_COL_SIZE)
            col_mask = cols[None, :] < N
            mask = row_mask and col_mask
            dy = tl.load(new_dY + cols[None, :], mask, other = 0.0).to(tl.float32)
            x = tl.load(new_X + cols[None, :], mask, other = 0.0).to(tl.float32)
            x_hat = (x - mean) * rstd
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
            wdy = dy * w
            dx_part2 += wdy
            dx_part3 += wdy * x_hat

        dx_2 = tl.sum(dx_part2, axis=1)[:, None]
        dx_3 = tl.sum(dx_part3, axis=1)[:, None]

        for off in range(0, N, BLOCK_COL_SIZE):
            cols = off + tl.arange(0, BLOCK_COL_SIZE)
            col_mask = cols[None, :] < N
            mask = row_mask and col_mask
            dy = tl.load(new_dY + cols[None, :], mask, other = 0.0).to(tl.float32)
            x = tl.load(new_X + cols[None, :], mask, other = 0.0).to(tl.float32)
            w = tl.load(W + cols, mask=cols < N, other = 0.0).to(tl.float32)
            x_hat = (x - mean) * rstd
            wdy = dy * w
            dx = rstd * (wdy - (dx_2 + x_hat * dx_3) / N)
            tl.store(new_DX + cols, dx.to(x.dtype), mask=mask)


def cfggen_wb_bw():
    block_m = [32, 64, 128, 512, 1024]
    block_n = [1, 4, 16, 32]
    num_stages = [1, 3]
    configs = [triton.Config({"BLOCK_ROW_SIZE": m, "BLOCK_COL_SIZE": n},
                             num_warps=1,
                             num_stages=s) for m in block_m for n in block_n for s in num_stages
    ]
    return configs

@libentry()
@triton.autotune(configs=cfggen_wb_bw(), key=["M", "N"])
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

    pid = tl.program_id(0)

    col_start = pid * BLOCK_COL_SIZE
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_COL_SIZE

    for col in range(col_start, N, step):
        col_off = col + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = col_off < N

        new_dY = dY + col_off
        new_X = X + col_off
        new_dW = dW + col_off
        new_dB = dB + col_off

        accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
        accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

        for off in range(0, M, BLOCK_ROW_SIZE):
            rows = off + tl.arange(0, BLOCK_ROW_SIZE)
            row_mask = rows[:, None] < M
            mask = row_mask and col_mask
            dy = tl.load(new_dY + rows[:, None] * N, mask, other = 0.0).to(tl.float32)
            x = tl.load(new_X + rows[:, None] * N, mask, other = 0.0).to(tl.float32)
            mean = tl.load(Mean + rows, mask = rows < M, other = 0.0)[:, None].to(tl.float32)
            rstd = tl.load(Rstd + rows, mask = rows < M, other = 0.0)[:, None].to(tl.float32)
            x_hat = (x - mean) * rstd
            accW += dy * x_hat
            accB += dy
        dw = tl.sum(accW, axis=0)
        db = tl.sum(accB, axis=0)
        tl.store(new_dW, dw[None, :], mask=col_mask)
        tl.store(new_dB, db[None, :], mask=col_mask)

class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                x,
                normalized_shape,
                weight,
                bias,
                eps=1e-5,
                cudnn_enable=True):
        logging.debug("GEMS LAYERNORM FORWARD")
        # dim = x.ndim - len(normalized_shape)
        # M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)
        M = x.numel() // N
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)
        mean = torch.empty(M, dtype=x.dtype, device=x.device)
        rstd = torch.empty(M, dtype=x.dtype, device=x.device)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]),)

        with torch.mlu.device(x.device):
            layer_norm_kernel[grid](x, y, weight, bias, mean, rstd, M, N, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.M = M
        ctx.N = N
        return y, mean, rstd

    @staticmethod
    def backward(ctx, out_grad, mean_grad, rstd_grad):
        logging.debug("GEMS LAYERNORM BACKWARD")
        out_grad = out_grad.contiguous()
        x, weight, mean, rstd = ctx.saved_tensors
        M, N = ctx.M, ctx.N
        if N <= MAX_C_MLU_LAYERNORM_BACKWARD:
            in_grad = torch.empty_like(out_grad)
            weight_grad = torch.zeros((weight.shape[0],), dtype=torch.float, device=weight.device)
            bias_grad = torch.zeros((weight.shape[0],), dtype=torch.float, device=weight.device)
            # enqueue kernel using forward pass heuristics
            # also compute partial sums for DW and DB
            grid = lambda META: (min(triton.cdiv(M, META['BLOCK_ROW_SIZE']), TOTAL_CORE_NUM),)
            with torch.mlu.device(x.device):
                layer_norm_backward_kernel[grid](
                    in_grad,
                    out_grad,
                    weight_grad,
                    bias_grad,
                    x,
                    weight,
                    mean,
                    rstd,
                    M,
                    N,
                    BLOCK_COL_SIZE=N,
                )
            weight_grad = weight_grad.to(x.dtype)
            bias_grad = bias_grad.to(x.dtype)
        else:
            in_grad = torch.empty_like(x)
            grid = lambda META: (min(triton.cdiv(M, META['BLOCK_ROW_SIZE']), TOTAL_CORE_NUM),)
            input_backward_kernel[grid](
                out_grad,
                x,
                weight,
                mean,
                rstd,
                in_grad,
                M,
                N,
            )
            weight_grad = torch.empty_like(weight)
            bias_grad = torch.empty_like(weight)
            grid = lambda META: (min(triton.cdiv(N, META['BLOCK_COL_SIZE']), TOTAL_CORE_NUM),)
            with torch.mlu.device(x.device):
                weight_bias_backward_kernel[grid](
                    out_grad,
                    x,
                    mean,
                    rstd,
                    weight_grad,
                    bias_grad,
                    M,
                    N,
                )

        return in_grad, None, weight_grad, bias_grad, None, None


def layer_norm(x, normalized_shape, weight, bias, eps=1e-5, cudnn_enable=True):
    return LayerNorm.apply(x, normalized_shape, weight, bias, eps,
                           cudnn_enable)
