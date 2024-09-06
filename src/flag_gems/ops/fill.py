import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.jit
def fill_kernel(
    out_ptr,
    N,
    value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE + cols
    tl.store(out_ptr + offset, value, mask=offset < N)


def fill(input, value):
    logging.debug("GEMS FILL")
    out = torch.empty_like(input)
    N = out.numel()
    BLOCK_SIZE = 512
    grid = triton.cdiv(N, BLOCK_SIZE)

    with torch.cuda.device(input.device):
        fill_kernel[grid,](out, N, value, BLOCK_SIZE)
    return out
