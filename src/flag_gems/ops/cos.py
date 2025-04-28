import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cos_func(x):
    return tl.cos(x.to(tl.float32))


@triton.jit
def cos_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr = 16,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.cos(x.to(tl.float32))
    tl.store(output_ptr + offsets, output, mask=mask)


def cos_block(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)

    # Flatten tensors for Triton kernel
    x_flat = x.contiguous()
    n_elements = output.numel()

    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    cos_kernel[grid](x_flat, output, n_elements, BLOCK_SIZE=16)

    return output


def cos(A):
    logging.debug("GEMS COS")
    return cos_block(A)
    # return cos_func(A)


def cos_(A):
    logging.debug("GEMS COS_")
    cos_func(A, out0=A)
    return A
