import logging

import torch
import triton
from rand import rand_kernel

from flag_gems.utils.random_utils import philox_cuda_seed_offset


def rand_like(size, *, dtype=None):
    logging.debug("GEMS RAND_LIKE")
    out = torch.empty_like(x)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    philox_seed, philox_offset = philox_cuda_seed_offset(N)
    rand_kernel[grid_fn](out, N, philox_seed, philox_offset, dtype)
    return out


if __name__ == "__main__":
    x = torch.randn(size=(2, 9), device="cuda")
    a = rand_like(x)
    print(a)
