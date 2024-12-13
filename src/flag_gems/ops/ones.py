import logging

import torch
import triton
import triton.language as tl

from ..runtime import device, torch_device_fn
from ..utils import libentry
from ..utils import triton_lang_extension as tle
from ..utils.shape_utils import volume

device_ = device


@libentry()
@triton.jit
def ones_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 1.0, mask=mask)


def ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logging.debug("GEMS ONES")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    with torch_device_fn.device(device):
        ones_kernel[grid](out, N, BLOCK_SIZE)
    return out
