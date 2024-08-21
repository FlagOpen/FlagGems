import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry
from .sum import sum


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": c, "BLOCK_D": d}, num_warps=4)
        for c in [256, 512, 1024]
        for d in [1, 4, 16]
    ],
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_fwd_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    w_tgt_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = not (tgt == ignore_index)

    w_ptrs = w_ptr + tgt
    w_tgt = tl.load(w_ptrs, mask=tgt_mask, other=0).to(tl.float32)
    w_tgt_ptrs = w_tgt_ptr + pid_n * D + offset_d
    tl.store(w_tgt_ptrs, w_tgt, mask=(tgt_mask & ignore_mask))

    inp_tgt_ptrs = inp_ptr + pid_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=tgt_mask, other=-float("inf")).to(tl.float32)
    out = inp_tgt * w_tgt * -1
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=(tgt_mask & ignore_mask))


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": c, "BLOCK_D": d}, num_warps=4)
        for c in [256, 512, 1024]
        for d in [1, 4, 16]
    ],
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "total_weight"])
def nll_loss_bwd_kernel(
    out_grad_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = (tgt != ignore_index)[None, :]

    w_ptrs = w_ptr + tgt
    w_tgt = tl.load(w_ptrs, mask=tgt_mask, other=0).to(tl.float32)[None, :]
    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = (tl.load(out_grad_ptrs, mask=tgt_mask, other=0)).to(tl.float32)[None, :]

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp_mask = inp_mask and offset_c[:, None] == tgt
        inp_grad = -1 * out_grad * w_tgt / total_weight
        inp_grad_ptrs = (
            inp_grad_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        )
        tl.store(inp_grad_ptrs, inp_grad.to(tl.float32), mask=(inp_mask & ignore_mask))


class NLLLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index):
        logging.debug("GEMS NLLLoss FWD")
        shape = list(inp.shape)
        dim = inp.ndim
        N = 1 if dim == 1 else shape[0]
        C = shape[0] if dim == 1 else shape[1]
        D = inp.numel() // N // C
        axis = 0 if dim == 1 else 1
        del shape[axis]

        assert ((i >= 0 and i < C) for i in target), "Target is out of bounds"
        assert list(target.shape) == shape, "Invalid target size"

        if weight is None:
            weight = torch.ones(
                [
                    C,
                ],
                dtype=inp.dtype,
                device=inp.device,
            )

        inp = inp.contiguous()
        tgt = target.contiguous()
        w = weight.contiguous()
        out = torch.zeros(shape, dtype=torch.float32, device=inp.device)
        w_tgt = torch.zeros(shape, dtype=torch.float32, device=inp.device)
        grid = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))

        with torch.cuda.device(inp.device):
            nll_loss_fwd_kernel[grid](inp, tgt, w, w_tgt, out, ignore_index, N, C, D)

        ctx.save_for_backward(inp, tgt, w)
        ctx.N = N
        ctx.C = C
        ctx.D = D
        ctx.ignore_index = ignore_index
        ctx.total_weight = 1
        ctx.shape = shape

        # redution: 0-None, 1-mean, 2-sum
        if reduction == 0:
            res = out.to(inp.dtype)
        elif reduction == 1:
            ctx.total_weight = sum(w_tgt).item()
            res = sum(out).to(inp.dtype) / ctx.total_weight
        else:
            res = sum(out).to(inp.dtype)

        return res

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS NLLLoss BWD")
        inp, tgt, w = ctx.saved_tensors
        print("tgt", tgt)
        N = ctx.N
        C = ctx.C
        D = ctx.D
        ignore_index = ctx.ignore_index
        total_weight = ctx.total_weight
        shape = ctx.shape

        out_grad = out_grad.broadcast_to(shape).contiguous()

        inp_grad = torch.zeros(inp.shape, dtype=inp.dtype, device=inp.device)
        grid = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))
        nll_loss_bwd_kernel[grid](
            out_grad, tgt, w, inp_grad, ignore_index, total_weight, N, C, D
        )

        return inp_grad, None, None, None, None, None


def nll_loss(inp, target, weight=None, reduction=1, ignore_index=-100):
    return NLLLoss.apply(inp, target, weight, reduction, ignore_index)
