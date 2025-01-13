import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("elu"), key=["M", "N"])
@triton.jit
def elu_forward_kernel(
    x_ptr,
    y_ptr,
    alpha,
    stride_xm,
    stride_ym,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)

    block_start_m = block_id_m * BLOCK_M
    block_start_n = block_id_n * BLOCK_N

    offsets_m = block_start_m + tl.arange(0, BLOCK_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_N)

    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(
        x_ptr + offsets_m[:, None] * stride_xm + offsets_n[None, :],
        mask=mask,
        other=0.0,
    )

    y = tl.where(x > 0, x, alpha * (tl.exp(x) - 1))

    tl.store(y_ptr + offsets_m[:, None] * stride_ym + offsets_n[None, :], y, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("elu"), key=["M", "N"])
@triton.jit
def elu_backward_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    alpha,
    stride_xm,
    stride_dym,
    stride_dxm,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_id_m = tl.program_id(0)
    block_id_n = tl.program_id(1)

    block_start_m = block_id_m * BLOCK_M
    block_start_n = block_id_n * BLOCK_N

    offsets_m = block_start_m + tl.arange(0, BLOCK_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_N)

    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(
        x_ptr + offsets_m[:, None] * stride_xm + offsets_n[None, :],
        mask=mask,
        other=0.0,
    )
    dy = tl.load(
        dy_ptr + offsets_m[:, None] * stride_dym + offsets_n[None, :],
        mask=mask,
        other=0.0,
    )

    dx = tl.where(x > 0, dy, alpha * tl.exp(x) * dy)

    tl.store(
        dx_ptr + offsets_m[:, None] * stride_dxm + offsets_n[None, :], dx, mask=mask
    )


class Elu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        assert x.is_cuda, "Input tensor must be on GPU"
        logging.debug("GEMS INDEX SELECT BACKWARD")
        assert x.ndim > 1, "dim can not be dimension 1 or 0"

        ctx.save_for_backward(x)
        ctx.alpha = alpha

        original_shape = x.shape
        x_2d = x.view(-1, original_shape[-1])
        y_2d = torch.empty_like(x_2d)

        M, N = x_2d.shape
        stride_xm = x_2d.stride(0)
        stride_ym = y_2d.stride(0)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        elu_forward_kernel[grid](
            x_2d,
            y_2d,
            alpha,
            stride_xm,
            stride_ym,
            M,
            N,
        )

        y = y_2d.view(original_shape)
        return y



def elu(x: torch.tensor, alpha: float = 1.0):
    return Elu.apply(x, alpha)
