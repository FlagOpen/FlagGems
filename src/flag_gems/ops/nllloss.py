import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry
from .cross_entropy_loss import sum_and_scale


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_2d_fwd_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    w_tgt_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_n

    if w_ptr is None:
        w_tgt = 1
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)
    tl.store(w_tgt_ptr + offsets_n, w_tgt, mask=mask_n)

    inp_tgt_ptrs = inp_ptr + offsets_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * w_tgt * -1
    tl.store(out_ptr + offsets_n, out, mask=mask_n)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_2d_bwd_kernel(
    out_grad_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    grad_stride,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_n

    if w_ptr is None:
        w_tgt = 1
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=mask_n, other=0).to(tl.float32)

    out_grad_ptrs = out_grad_ptr + offsets_n * grad_stride
    out_grad = tl.load(out_grad_ptrs, mask=mask_n, other=0).to(tl.float32)

    inp_grad = tl.where(ignore_mask, -1 * out_grad * w_tgt / total_weight, 0)
    inp_grad_ptrs = inp_grad_ptr + offsets_n * C + tgt
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_n)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_multi_fwd_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    w_tgt_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    D,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = (offset_nd % D)[None, :]
    offset_n = (offset_nd // D)[:, None]

    mask_block = offset_n < N and offset_d < D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_block

    if w_ptr is None:
        w_tgt = 1
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)
    w_tgt_ptrs = w_tgt_ptr + offset_n * D + offset_d
    tl.store(w_tgt_ptrs, w_tgt, mask=mask_block)

    inp_tgt_ptrs = inp_ptr + offset_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * w_tgt * -1
    out_ptrs = out_ptr + offset_n * D + offset_d
    tl.store(out_ptrs, out, mask=mask_block)


@libentry()
@triton.jit(do_not_specialize=["ignore_index", "total_weight"])
def nll_loss_multi_bwd_kernel(
    out_grad_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    D,
    grad_stride_n,
    grad_stride_d,
    BLOCK_ND: tl.constexpr = 128,
):
    pid_nd = tl.program_id(0)
    offset_nd = pid_nd * BLOCK_ND + tl.arange(0, BLOCK_ND)
    offset_d = offset_nd % D[None, :]
    offset_n = offset_nd // D[:, None]

    mask_block = offset_n < N and offset_d < D

    tgt_ptrs = tgt_ptr + offset_n * D + offset_d
    tgt = tl.load(tgt_ptrs, mask=mask_block, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_block

    if w_ptr is None:
        w_tgt = 1
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=mask_block, other=0).to(tl.float32)

    out_grad_ptrs = out_grad_ptr + offset_n * grad_stride_n + offset_d * grad_stride_d
    out_grad = tl.load(out_grad_ptrs, mask=mask_block, other=0).to(tl.float32)

    inp_grad = tl.where(ignore_mask, -1 * out_grad * w_tgt / total_weight, 0)
    inp_grad_ptrs = inp_grad_ptr + offset_n * C * D + tgt * D + offset_d
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_block)


# Negative Log Likelihood Loss (NLLLoss)
#
# This loss function is used for training classification problems with C classes.
#
# Parameters:
# - input (Tensor):
#   - Expected to contain log-probabilities for each class.
#   - Shape can be either:
#     - (minibatch, C) for standard classification tasks.
#     - (minibatch, C, d1, d2, ..., dK) for K-dimensional inputs (e.g., per-pixel loss for 2D images).
#
# - target (Tensor):
#   - Should contain class indices in the range [0, C-1].
#   - If ignore_index is specified, this index can be outside the class range
#       and will be ignored in the loss computation.
#
# - weight (1D Tensor, optional):
#   - Assigns weight to each class, useful for unbalanced datasets.
#
# Reduction modes:
# - 'none': returns per-sample loss (shape: (N,)).
# - 'mean' (default): computes the mean of the weighted losses.
# - 'sum': computes the sum of the weighted losses.
#
# Mathematical description:
# - Unreduced loss:
#   l_n = -w_y_n * x_n, where w_c = weight[c] * 1{c != ignore_index}.
# - Reduced loss (depending on the specified reduction mode):
#   - mean: ℓ(x, y) = (1/N) * Σ(w_y_n * l_n)
#   - sum: ℓ(x, y) = Σ(l_n)
class NegativeLogLikeLoss(torch.autograd.Function):
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

        assert list(target.shape) == shape, "Invalid target size"
        assert inp.ndim >= 1, "Invalid input ndim"

        inp = inp.contiguous()
        tgt = target.contiguous()
        w = None if weight is None else weight.contiguous()
        out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
        w_tgt = torch.empty(shape, dtype=inp.dtype, device=inp.device)

        if inp.ndim > 2:
            grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
            with torch.cuda.device(inp.device):
                nll_loss_multi_fwd_kernel[grid](
                    inp, tgt, w, w_tgt, out, ignore_index, N, C, D
                )
        else:
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
            with torch.cuda.device(inp.device):
                nll_loss_2d_fwd_kernel[grid](
                    inp, tgt, w, w_tgt, out, ignore_index, N, C
                )

        # redution: 0-None, 1-mean, 2-sum
        if reduction == 1:
            res = torch.empty([], dtype=inp.dtype, device=inp.device)
            wgt_sum = torch.empty([], dtype=inp.dtype, device=inp.device)
            sum_and_scale[(1,)](out, res, N * D, True, scale=w_tgt, mean_num=wgt_sum)
            out = res
        elif reduction == 2:
            res = torch.empty([], dtype=inp.dtype, device=inp.device)
            sum_and_scale[(1,)](out, res, N * D, False)
            out = res

        if inp.requires_grad:
            ctx.save_for_backward(inp, tgt, w)
            ctx.N = N
            ctx.C = C
            ctx.D = D
            ctx.ignore_index = ignore_index
            ctx.total_weight = wgt_sum if reduction == 1 else 1
            ctx.shape = shape

        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS NLLLoss BWD")
        inp, tgt, w = ctx.saved_tensors
        N = ctx.N
        C = ctx.C
        D = ctx.D
        ignore_index = ctx.ignore_index
        total_weight = (
            ctx.total_weight.item()
            if isinstance(ctx.total_weight, torch.Tensor)
            else ctx.total_weight
        )
        shape = ctx.shape

        out_grad = out_grad.contiguous().broadcast_to(shape)
        inp_grad = torch.zeros_like(inp)

        if inp.ndim > 2:
            grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
            nll_loss_multi_bwd_kernel[grid](
                out_grad,
                tgt,
                w,
                inp_grad,
                ignore_index,
                total_weight,
                N,
                C,
                D,
                out_grad.stride()[0],
                out_grad.stride()[-1],
            )
        else:
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
            nll_loss_2d_bwd_kernel[grid](
                out_grad,
                tgt,
                w,
                inp_grad,
                ignore_index,
                total_weight,
                N,
                C,
                out_grad.stride()[-1],
            )

        return inp_grad, None, None, None, None, None


def nll_loss(inp, target, weight=None, reduction=1, ignore_index=-100):
    return NegativeLogLikeLoss.apply(inp, target, weight, reduction, ignore_index)
