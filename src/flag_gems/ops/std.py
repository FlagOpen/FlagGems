import logging

import torch
import triton
import triton.language as tl

from .sqrt import sqrt
from .var_mean import var_mean

logger = logging.getLogger(__name__)


def std(x, dim=None, unbiased=True, keepdim=False):
    logger.debug("GEMS STD Forward")

    dim_list = (
        dim if isinstance(dim, (list, tuple)) else ([dim] if dim is not None else None)
    )

    variance, _ = var_mean(x, dim=dim_list, unbiased=unbiased, keepdim=keepdim)
    std_dev = sqrt(variance)
    return std_dev


@triton.jit
def _std_backward_kernel(
    grad_x_ptr,
    grad_output_ptr,
    x_ptr,
    mean_ptr,
    std_dev_ptr,
    divisor,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elements
    grad_output = tl.load(grad_output_ptr + offset, mask=mask, other=0.0)
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offset, mask=mask, other=0.0)
    std_dev = tl.load(std_dev_ptr + offset, mask=mask, other=0.0)
    std_dev_safe = tl.where(std_dev > 0, std_dev, 1.0)
    grad_output_f32 = grad_output.to(tl.float32)
    x_f32 = x.to(tl.float32)
    mean_f32 = mean.to(tl.float32)
    std_dev_safe_f32 = std_dev_safe.to(tl.float32)
    divisor_f32 = divisor.to(tl.float32)
    grad_x = grad_output_f32 * (x_f32 - mean_f32) / (std_dev_safe_f32 * divisor_f32)
    tl.store(grad_x_ptr + offset, grad_x, mask=mask)


def std_backward(grad, x, mean, std, unbiased):
    logger.debug("GEMS STD Backward")

    if mean.numel() == 1:
        num_elements_reduced = x.numel()
    else:
        dim_list = [
            i for i, (x_s, m_s) in enumerate(zip(x.shape, mean.shape)) if x_s != m_s
        ]

        if not dim_list and x.shape != mean.shape:
            dim_list = list(range(x.ndim))

        num_elements_reduced = 1
        for d in dim_list:
            num_elements_reduced *= x.shape[d]

    divisor = num_elements_reduced - 1.0 if unbiased else num_elements_reduced

    grad_x = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(x.numel(), META["BLOCK_SIZE"]),)

    _std_backward_kernel[grid](
        grad_x,
        grad.expand_as(x),
        x,
        mean.expand_as(x),
        std.expand_as(x),
        divisor,
        x.numel(),
    )
    return grad_x
