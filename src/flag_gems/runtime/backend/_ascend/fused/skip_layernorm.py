import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel(
        Y,  # pointer to the output
        X,  # pointer to the input
        R,  # pointer to the residual
        W,  # pointer to the weights
        B,  # pointer to the biases
        y_stride_r,
        y_stride_c,
        x_stride_r,  # how much to increase the pointer when moving by 1 row
        x_stride_c,  # how much to increase the pointer when moving by 1 col
        r_stride_r,  # how much to increase the pointer when moving by 1 row
        r_stride_c,  # how much to increase the pointer when moving by 1 col
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r
    R += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)

    x += r

    mean = tl.sum(x, axis=0) / N

    # Compute variance
    _var = tl.where(mask, x - mean, 0.0)
    _var = _var * _var
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)

    x_hat = (x - mean) * rstd
    y = w * x_hat + b
    y = y.to(Y.dtype.element_ty)
    tl.store(Y + cols * y_stride_c, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def skip_layer_norm_kernel_1(
        Y,  # pointer to the output
        X,  # pointer to the input
        R,  # pointer to the residual
        W,  # pointer to the weights
        B,  # pointer to the biases
        y_stride_r,
        y_stride_c,
        x_stride_r,  # how much to increase the pointer when moving by 1 row
        x_stride_c,  # how much to increase the pointer when moving by 1 col
        r_stride_r,  # how much to increase the pointer when moving by 1 row
        r_stride_c,  # how much to increase the pointer when moving by 1 col
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    loops = tl.cdiv(N, BLOCK_SIZE)
    sum_x = tl.zeros((), dtype=tl.float32)  # 显式指定为float32
    X += pid * x_stride_r
    R += pid * r_stride_r
    Y += pid * y_stride_r
    # 这个切分比较特别， 需要一次load进来整行的数据再进行计算，所以需要传N进来
    # 分块读取整维数据
    for process in range(loops):
        cols = process * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        sum_x += tl.sum(x, axis=0)
    # 计算整体均值
    mean = sum_x / N
    sum_var = tl.zeros((), dtype=tl.float32)  # 显式指定为float32
    # 分块计算标准差
    for process in range(loops):
        cols = process * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        # Compute variance
        _var = tl.where(mask, x - mean, 0.0)
        _var = _var * _var
        sum_var += tl.sum(_var, axis=0)
    # 计算整体标准差
    var = sum_var / N
    rstd = 1 / tl.sqrt(var + eps)
    # 分块计算结果
    for process in range(loops):
        cols = process * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * r_stride_c, mask, other=0.0).to(tl.float32)
        x += r
        x_hat = (x - mean) * rstd
        y = w * x_hat + b
        y = y.to(Y.dtype.element_ty)
        tl.store(Y + cols * y_stride_c, y, mask=mask)


class SkipLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, normalized_shape, weight, bias, eps=1e-5):
        logging.debug("GEMS_ASCEND SKIP LAYERNORM FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = min(math.prod(x.shape[:dim]), 65535)
        N = math.prod(normalized_shape)
        BLOCK_SIZE = min(triton.next_power_of_2(N), 4096)
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        y = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            skip_layer_norm_kernel_1[M,](
                y, x, residual, weight, bias, N, 1, N, 1, N, 1, N, eps, BLOCK_SIZE
            )
        return y


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    return SkipLayerNorm.apply(x, residual, normalized_shape, weight, bias, eps)
