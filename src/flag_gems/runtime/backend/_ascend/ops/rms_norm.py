# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Ascend limits the product of all launch-grid dimensions.
_ASCEND_MAX_CORE_DIM = 65535


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,  # pointer to the output
    INV_RMS,  # pointer to inverse rms
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    for pid in range(ext.program_id(0), M, ext.num_programs(0)):
        var = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(
                X + pid * x_stride_r + cols * x_stride_c, mask, other=0.0
            ).to(tl.float32)
            var += tl.sum(x * x / N)

        rrms = 1 / tl.sqrt(var + eps)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(
                X + pid * x_stride_r + cols * x_stride_c, mask, other=0.0
            ).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0)
            y = (x * rrms).to(Y.dtype.element_ty) * w
            tl.store(Y + pid * y_stride_r + cols * y_stride_c, y, mask=mask)

        tl.store(INV_RMS + pid, rrms)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DX,  # pointer to the output
    W,  # pointer to the weights
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    for pid in range(ext.program_id(0), M, ext.num_programs(0)):
        inv_rms = tl.load(INV_RMS + pid).to(tl.float32)

        row_sum_stats = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(
                X + pid * x_stride_r + cols * x_stride_c, mask, other=0.0
            ).to(tl.float32)
            dy = tl.load(
                DY + pid * x_stride_r + cols * x_stride_c, mask, other=0.0
            ).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
            dy = dy * w
            normalized_buf = x * inv_rms
            row_sum_stats += tl.sum(normalized_buf * dy)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(
                X + pid * x_stride_r + cols * x_stride_c, mask, other=0.0
            ).to(tl.float32)
            dy = tl.load(
                DY + pid * x_stride_r + cols * x_stride_c, mask, other=0.0
            ).to(tl.float32)
            w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
            dy = dy * w
            normalized_buf = x * inv_rms
            norm_val = normalized_buf / N
            dx = (dy - norm_val * row_sum_stats) * inv_rms
            tl.store(DX + pid * dx_stride_r + cols * dx_stride_c, dx, mask=mask)


@libentry()
@triton.jit
def rms_norm_grad_dw_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DW,  # pointer to the output
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    col_blocks = tl.cdiv(N, COL_BLOCK_SIZE)
    tiles = tl.cdiv(M, ROW_BLOCK_SIZE) * col_blocks
    for pid in range(ext.program_id(0), tiles, ext.num_programs(0)):
        row_pid = pid // col_blocks
        col_pid = pid % col_blocks

        row_start = row_pid * ROW_BLOCK_SIZE
        col_start = col_pid * COL_BLOCK_SIZE

        offset = row_start * x_stride_r + col_start * x_stride_c

        rows = tl.arange(0, ROW_BLOCK_SIZE)
        cols = tl.arange(0, COL_BLOCK_SIZE)

        row_mask = (row_start + rows) < M
        col_mask = (col_start + cols) < N

        x = tl.load(
            X + offset + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
            row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        inv_rms = tl.load(
            INV_RMS + row_start + rows, row_mask, other=0.0
        ).to(tl.float32)
        dy = tl.load(
            DY + offset + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
            row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        d_weight = x * dy * inv_rms[:, None]
        partial_dweight_sum = tl.sum(d_weight, axis=0)

        tl.store(
            DW + row_pid * N + col_start + cols,
            partial_dweight_sum,
            mask=col_mask,
        )


def rms_norm_forward(x, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_ASCEND RMS_NORM_FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

    if M:
        # Power-of-two tiles keep both tl.arange and Ascend UB usage bounded.
        block_size = min(triton.next_power_of_2(N), 4096)
        with torch_device_fn.device(x.device):
            rms_norm_kernel[(min(M, _ASCEND_MAX_CORE_DIM),)](
                y, inv_rms, x, weight, N, 1, N, 1, M, N, eps, block_size
            )
    return y, inv_rms


def rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps=1e-5):
    logger.debug("GEMS_ASCEND RMS_NORM_BACKWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    x = x.contiguous()
    dy = dy.contiguous()
    inv_rms = inv_rms.contiguous()
    weight = weight.contiguous()
    dx = torch.empty_like(x)
    if not M:
        return dx, torch.zeros_like(weight)

    block_size = min(triton.next_power_of_2(N), 1024)
    row_block_size = 16
    col_block_size = 256
    row_block_num = triton.cdiv(M, row_block_size)
    col_block_num = triton.cdiv(N, col_block_size)
    partial_buffer = torch.empty(
        (row_block_num, N), dtype=torch.float32, device=x.device
    )

    with torch_device_fn.device(x.device):
        rms_norm_grad_dx_kernel[(min(M, _ASCEND_MAX_CORE_DIM),)](
            x, dy, inv_rms, dx, weight, N, 1, N, 1, M, N, eps, block_size
        )
        # Flatten before capping: limiting each axis separately is insufficient.
        grid = (min(row_block_num * col_block_num, _ASCEND_MAX_CORE_DIM),)
        rms_norm_grad_dw_kernel[grid](
            x,
            dy,
            inv_rms,
            partial_buffer,
            N,
            1,
            N,
            1,
            M,
            N,
            row_block_size,
            col_block_size,
        )
        dw = torch.sum(partial_buffer, dim=0, dtype=torch.float32)
        dw = dw.to(weight.dtype).reshape(weight.shape)
    return dx, dw


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        y, inv_rms = rms_norm_forward(x, normalized_shape, weight, eps)
        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, inv_rms, weight = ctx.saved_tensors
        dx, dw = rms_norm_backward(
            dy, x, inv_rms, ctx.normalized_shape, weight, ctx.eps
        )
        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
