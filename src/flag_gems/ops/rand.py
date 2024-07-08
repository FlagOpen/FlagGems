import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_cuda_seed_offset
from flag_gems.utils.shape_utils import volume


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=[
        "N",
    ],
)
@triton.jit
def rand_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    dtype,
    N_BLOCK_SIZE: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl_rand_dtype)
    pid = tl.program_id(0) * N_BLOCK_SIZE
    offset = pid + tl.arange(0, N_BLOCK_SIZE)
    mask = offset < N
    philox_offset = philox_offset + offset
    out = tl.rand(philox_seed, philox_offset, n_rounds=6)
    tl.store(out_ptr + offset, out, mask=mask)


def rand(size, *, dtype=None):
    logging.debug("GEMS RAND")
    out = torch.empty(size, dtype=dtype, device=torch.device("cuda"))
    N = volume(size)
    grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
    philox_seed, philox_offset = philox_cuda_seed_offset(N)
    rand_kernel[grid_fn](out, N, philox_seed, philox_offset, dtype)
    return out


if __name__ == "__main__":
    try:
        tl_rand_dtype = tl.int64

        @triton.jit
        def _rand(seed, offset):
            offset = offset.to(tl_rand_dtype)

        _grid = (1,)
        _seed, _offset = philox_cuda_seed_offset(0)
        _rand[_grid](_seed, _offset)
    except Exception:
        tl_rand_dtype = tl.int32

    del _grid
    del _seed
    del _offset
    a = rand(size=(10, 2))
    print(a)
