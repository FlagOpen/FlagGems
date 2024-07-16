import logging

import torch
import torch_mlu
import triton
import triton.language as tl
from ..utils.random_utils import philox_mlu_seed_offset, uint_to_uniform_float
from ..utils import libentry, TOTAL_CORE_NUM


def heur_block(args):
    if args["N"] <= 512:
        return 512
    else:
        return 1024


def heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


@triton.heuristics(
    {
        "BLOCK": heur_block,
    }
)
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_forward_kernel(
    X,
    Y,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4  # philox generate 128 random bits at a time
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)


    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    i4_start = pid * BLOCK
    block_start = pid * UNROLL * BLOCK
    step = num_jobs * BLOCK * UNROLL
    mp = 1.0 / (1.0 - p)

    r = tl.empty([4, BLOCK], dtype=tl.float32)
    for block_offset in range(block_start, N, step):
        c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
        i4 = i4_start + tl.arange(0, BLOCK)
        c0 += i4
        _O = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
        r[0, :] = uint_to_uniform_float(r0)
        r[1, :] = uint_to_uniform_float(r1)
        r[2, :] = uint_to_uniform_float(r2)
        r[3, :] = uint_to_uniform_float(r3)

        mask = r > p

        off = block_offset + tl.arange(0, UNROLL * BLOCK)
        x = tl.load(X + off, mask=off < N, other=0.0)

        y = x * mp * tl.reshape(mask, [UNROLL * BLOCK], can_reorder=True)  # tl.where(mask0, x0 * p, 0.0)
        tl.store(Y + off, y, mask=off < N)
        i4_start += num_jobs * BLOCK


@triton.heuristics(
    {
        "BLOCK": heur_block,
    }
)
@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_backward_kernel(
    DY,
    DX,
    N,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    i4_start = pid * BLOCK
    block_start = pid * UNROLL * BLOCK
    step = num_jobs * BLOCK * UNROLL
    mp = 1.0 / (1.0 - p)

    r = tl.empty([4, BLOCK], dtype=tl.float32)
    for block_offset in range(block_start, N, step):
        c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
        i4 = i4_start + tl.arange(0, BLOCK)
        c0 += i4
        _O = c0 * 0
        r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
        r[0, :] = uint_to_uniform_float(r0)
        r[1, :] = uint_to_uniform_float(r1)
        r[2, :] = uint_to_uniform_float(r2)
        r[3, :] = uint_to_uniform_float(r3)

        mask = r > p
        off = block_offset + tl.arange(0, UNROLL * BLOCK)

        dy = tl.load(DY + off, mask=off < N, other=0.0)

        dx = mp * dy * tl.reshape(mask, [UNROLL * BLOCK], can_reorder=True)

        tl.store(DX + off, dx, mask=off < N)
        i4_start += num_jobs * BLOCK


UNROLL = 4


class NativeDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, train):
        logging.debug("GEMS NATIVE DROPOUT FORWARD")
        assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
        device = x.device
        x = x.contiguous()
        out = torch.empty_like(x)
        N = x.numel()
        grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),)
        # (TODO) Using Triton autotuner makes kernel parameters opaque to the caller,
        # hence we cannot obtain the per thread offset as in Pytorch.
        increment = triton.cdiv(N, UNROLL)
        with torch.mlu.device(device):
            philox_seed, philox_offset = philox_mlu_seed_offset(increment)
            dropout_forward_kernel[grid_fn](x, out, N, p, philox_seed, philox_offset, num_warps=1, num_stages=3)
        ctx.p = p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        return out, None

    @staticmethod
    def backward(ctx, grad_outputs, kwargs):
        logging.debug("GEMS NATIVE DROPOUT BACKWARD")
        device = grad_outputs.device
        grad_outputs = grad_outputs.contiguous()
        grad_inputs = torch.empty_like(grad_outputs)
        N = grad_outputs.numel()
        grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK"] * UNROLL), TOTAL_CORE_NUM),)
        with torch.mlu.device(device):
            dropout_backward_kernel[grid_fn](
                grad_outputs, grad_inputs, N, ctx.p, ctx.philox_seed, ctx.philox_offset, num_stages=3, num_warps=1
            )
        return grad_inputs, None, None


def native_dropout(x, p=0.5, train=True):
    return NativeDropout.apply(x, p, train)
