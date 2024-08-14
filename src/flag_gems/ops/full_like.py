import logging

import torch
import triton

from .full import full_kernel


def full_like(
    x,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    logging.debug("GEMS FULL_LIKE")
    if device is None:
        device = x.device
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    BLOCK_SIZE = triton.cdiv(N, 8)  # CLUSTER_NUM=8  empty has dirty data
    with torch.cuda.device(x.device):
        full_kernel[grid_fn](out, N, fill_value, BLOCK_SIZE=BLOCK_SIZE)
    return out
