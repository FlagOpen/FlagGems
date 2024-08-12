import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.random_utils import philox_mlu_seed_offset, uint_to_uniform_float
from flag_gems.utils.shape_utils import volume
from ..utils import TOTAL_CORE_NUM

try:
    pair_uniform_to_normal = tl.pair_uniform_to_normal
except AttributeError:

    @triton.jit
    def pair_uniform_to_normal(u1, u2):
        """Box-Muller transform"""
        u1 = tl.maximum(1.0e-7, u1)
        th = 6.283185307179586 * u2
        r = tl.sqrt(-2.0 * tl.log(u1))
        return r * tl.cos(th), r * tl.sin(th)


def heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024

@triton.heuristics(
    {
        "BLOCK": heur_block,
    }
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randn_kernel(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    i4_start = pid * BLOCK
    i4_step = num_jobs * BLOCK

    block_start = pid * BLOCK * UNROLL
    step = num_jobs * BLOCK * UNROLL

    res = tl.empty([UNROLL, BLOCK], dtype=tl.float32)
    for block_offset in range(block_start, N, step):
        c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
        i4 = i4_start + tl.arange(0, BLOCK)
        c0 += i4
        _O = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
        r0 = uint_to_uniform_float(r0)
        r1 = uint_to_uniform_float(r1)
        r2 = uint_to_uniform_float(r2)
        r3 = uint_to_uniform_float(r3)

        r0 = tl.maximum(1.0e-7, r0)
        th = 6.283185307179586 * r1
        r = tl.sqrt(-2.0 * tl.log(r0))
        res[0, :] = r * tl.cos(th)
        res[1, :] = r * tl.sin(th)

        r2 = tl.maximum(1.0e-7, r2)
        th = 6.283185307179586 * r3
        r = tl.sqrt(-2.0 * tl.log(r2))
        res[2, :] = r * tl.cos(th)
        res[3, :] = r * tl.sin(th)

        off = block_offset + tl.arange(0, BLOCK * UNROLL)
        tl.store(out_ptr + off, tl.reshape(res, [BLOCK * UNROLL], can_reorder=True), mask=off < N)
        i4_start += i4_step

UNROLL = 4

def randn(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logging.debug("GEMS RANDN")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("mlu")

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),)
    philox_seed, philox_offset = philox_mlu_seed_offset(N)
    with torch.cuda.device(device):
        randn_kernel[grid_fn](out, N, philox_seed, philox_offset, num_stages=3, num_warps=1)
    return out
