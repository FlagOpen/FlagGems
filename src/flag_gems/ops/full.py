import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.shape_utils import volume
from ..utils import TOTAL_CORE_NUM

@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_stages=1, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 4096}, num_stages=1, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 16384}, num_stages=1, num_warps=1),
        triton.Config(kwargs={'BLOCK_SIZE': 65536}, num_stages=1, num_warps=1),
    ],
    key=['n_elements'],
)
@triton.jit(do_not_specialize=["fill_value"])
def full_kernel(
    output_ptr,
    n_elements,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, fill_value, mask=mask)


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    logging.debug("GEMS FULL")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cuda")

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch.cuda.device(device):
        full_kernel[grid_fn](out, N, fill_value)
    return out
