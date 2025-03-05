import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

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
    grid_fn = (12, 1, 1)
    block_size = triton.next_power_of_2(triton.cdiv(N, 12))
    with torch_device_fn.device(device):
        ones_kernel[grid_fn](out, N, BLOCK_SIZE=block_size)
    return out
