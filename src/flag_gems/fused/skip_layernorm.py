import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry


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
def skip_layer_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = row < M
    Y += row * N
    X += row * N
    R += row * N

    # Compute mean
    _mean = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    # Compute variance
    _var = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        _mean += x
        _var += x * x
    mean = tl.sum(_mean, axis=1) / N
    mean_bc = mean[:, None]
    # Compute variance
    var = tl.sum(_var, axis=1) / N - (mean * mean)
    var = var[:, None]
    rstd = 1 / tl.sqrt(var + eps)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        w = tl.load(W + cols, col_mask)
        b = tl.load(B + cols, col_mask)
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols, mask, other=0.0).to(tl.float32)
        x += r
        x = tl.where(col_mask, x - mean_bc, 0.0)
        x_hat = x * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


class SkipLayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logging.debug("GEMS SKIP LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_ROW_SIZE"]), )
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)

        with torch.mlu.device(x.device):
            skip_layer_norm_kernel[grid](y, x, residual, weight, bias, M, N,
                                         eps)
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias,
                               eps)
