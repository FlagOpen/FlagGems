import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry
from .cross_entropy_loss import sum_and_scale


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_forward_kernel(
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
        w_tgt = ignore_mask.to(tl.float32)
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)
    tl.store(w_tgt_ptr + offsets_n, w_tgt, mask=mask_n)

    inp_tgt_ptrs = inp_ptr + offsets_n * C + tgt
    inp_tgt = tl.load(inp_tgt_ptrs, mask=ignore_mask, other=0).to(tl.float32)
    out = inp_tgt * w_tgt * -1
    tl.store(out_ptr + offsets_n, out, mask=mask_n)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    reduction: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offsets_n < N

    tgt = tl.load(tgt_ptr + offsets_n, mask=mask_n, other=0)
    ignore_mask = not (tgt == ignore_index) and mask_n

    if w_ptr is None:
        w_tgt = ignore_mask
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offsets_n
        out_grad = tl.load(out_grad_ptrs, mask=mask_n, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    total_w = tl.load(total_weight).to(tl.float32)
    inp_grad = tl.where(ignore_mask, -1 * out_grad * w_tgt / total_w, 0)
    inp_grad_ptrs = inp_grad_ptr + offsets_n * C + tgt
    tl.store(inp_grad_ptrs, inp_grad, mask=mask_n)


@libentry()
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_forward_kernel(
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
        w_tgt = ignore_mask.to(tl.float32)
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
@triton.jit(do_not_specialize=["ignore_index"])
def nll_loss2d_backward_kernel(
    out_grad_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    total_weight,
    N,
    C,
    D,
    reduction: tl.constexpr = 1,
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
        w_tgt = ignore_mask
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=ignore_mask, other=0).to(tl.float32)

    if reduction == 0:
        out_grad_ptrs = out_grad_ptr + offset_n * D + offset_d
        out_grad = tl.load(out_grad_ptrs, mask=mask_block, other=0).to(tl.float32)
    else:
        out_grad = tl.load(out_grad_ptr).to(tl.float32)

    total_w = tl.load(total_weight).to(tl.float32)
    inp_grad = tl.where(ignore_mask, -1 * out_grad * w_tgt / total_w, 0)
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


# 1d & 2d tensor
def nll_loss_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logging.debug("GEMS NLL Loss FWD")
    assert self.ndim <= 2, "Invalid input ndim"
    shape = list(target.shape)
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]
    assert target.numel() == N, "Invalid target size"

    self = self.contiguous()
    target = target.contiguous()
    w = None if weight is None else weight.contiguous()

    out = torch.empty(shape, dtype=self.dtype, device=self.device)
    w_tgt = torch.empty(shape, dtype=self.dtype, device=self.device)
    total_weight = torch.ones([], dtype=self.dtype, device=self.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    with torch.cuda.device(self.device):
        nll_loss_forward_kernel[grid](self, target, w, w_tgt, out, ignore_index, N, C)

    # redution: 0-None, 1-mean, 2-sum
    if reduction == 1:
        output = torch.empty([], dtype=self.dtype, device=self.device)
        sum_and_scale[(1,)](out, output, N, True, scale=w_tgt, mean_num=total_weight)
    elif reduction == 2:
        output = torch.empty([], dtype=self.dtype, device=self.device)
        sum_and_scale[(1,)](out, output, N, False)
    else:
        output = out

    return output, total_weight


def nll_loss_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logging.debug("GEMS NLL Loss BWD")
    N = 1 if self.ndim == 1 else self.shape[0]
    C = self.shape[-1]

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    nll_loss_backward_kernel[grid](
        grad_output,
        target,
        weight,
        grad_input,
        ignore_index,
        total_weight,
        N,
        C,
        reduction,
    )

    return grad_input


# 3d+ tensor
def nll_loss2d_forward(self, target, weight=None, reduction=1, ignore_index=-100):
    logging.debug("GEMS NLL Loss2d FWD")
    assert self.ndim == 4, "Invalid input ndim"

    shape = list(target.shape)
    N, C, _, D = self.shape
    assert shape == [N, 1, D], "Invalid target size"

    self = self.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    out = torch.empty(shape, dtype=self.dtype, device=self.device)
    w_tgt = torch.empty(shape, dtype=self.dtype, device=self.device)
    total_weight = torch.ones([], dtype=self.dtype, device=self.device)

    grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
    with torch.cuda.device(self.device):
        nll_loss2d_forward_kernel[grid](
            self, target, weight, w_tgt, out, ignore_index, N, C, D
        )

    # redution: 0-None, 1-mean, 2-sum
    if reduction == 1:
        output = torch.empty([], dtype=self.dtype, device=self.device)
        sum_and_scale[(1,)](
            out, output, N * D, True, scale=w_tgt, mean_num=total_weight
        )
    elif reduction == 2:
        output = torch.empty([], dtype=self.dtype, device=self.device)
        sum_and_scale[(1,)](out, output, N * D, False)
    else:
        output = out

    return output, total_weight


def nll_loss2d_backward(
    grad_output,
    self,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    total_weight=None,
):
    logging.debug("GEMS NLL Loss2d BWD")
    N, C, _, D = self.shape

    grad_output = grad_output.contiguous()
    target = target.contiguous()
    weight = None if weight is None else weight.contiguous()

    grad_input = torch.zeros_like(self).contiguous()

    grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_ND"]),)
    nll_loss2d_backward_kernel[grid](
        grad_output,
        target,
        weight,
        grad_input,
        ignore_index,
        total_weight,
        N,
        C,
        D,
        reduction,
    )

    return grad_input
