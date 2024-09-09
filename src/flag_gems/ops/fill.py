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
    IS_VALUE_SCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE + cols
    if not IS_VALUE_SCALAR:
        value_scalar = tl.load(value)  # load the value from the tensor.
    else:
        value_scalar = value  # value is float scalar.

    tl.store(out_ptr + offset, value_scalar, mask=offset < N)


def fill(input, value):
    logging.debug("GEMS FILL")
    out = torch.empty_like(input)
    N = out.numel()
    BLOCK_SIZE = 512
    grid = triton.cdiv(N, BLOCK_SIZE)

    with torch.cuda.device(input.device):
        if isinstance(value, torch.Tensor):
            IS_VALUE_SCALAR = False
        else:
            IS_VALUE_SCALAR = True
        fill_kernel[grid,](out, N, value, BLOCK_SIZE, IS_VALUE_SCALAR)
    return out
