import logging

import torch
import triton
import triton.language as tl


logger = logging.getLogger(__name__)

@triton.jit
def celu_forward_kernel(
    x_ptr, output_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.where(
        x > 0, x, alpha * (tl.exp(x / alpha) - 1),        
    )
    tl.store(output_ptr + offsets, output, mask=mask)

def celu(A, alpha=1.0):
    logger.debug("GEMS CELU")
    n_elements = A.numel()
    output = torch.empty_like(A)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    celu_forward_kernel[grid](
        A, output, alpha, n_elements, BLOCK_SIZE=2048
    )
    return output

def celu_(A, alpha=1.0):
    logger.debug("GEMS CELU_")
    n_elements = A.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    celu_forward_kernel[grid](
        A, A, alpha, n_elements, BLOCK_SIZE=2048
    )
    return A



