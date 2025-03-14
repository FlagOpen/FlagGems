import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def celoss_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    reduction: tl.constexpr,
    ignore_index: tl.constexpr,
    label_smoothing: tl.constexpr,
):
    return tl.cross_entropy_loss(inp_ptr, tgt_ptr, w_ptr, reduction, ignore_index, label_smoothing)


@triton.jit
def celoss_bwd_kernel(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    reduction: tl.constexpr,
    ignore_index: tl.constexpr,
    label_smoothing: tl.constexpr,
):
    return tl.cross_entropy_loss_bwd(out_grad_ptr, inp_ptr, tgt_ptr, w_ptr, reduction, ignore_index, label_smoothing)



class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index, label_smoothing):
        logging.debug("GEMS CrossEntropyLoss")
        # label_smoothing not supported

        shape = list(inp.shape)
        dim = inp.ndim
        C = shape[0] if dim == 1 else shape[1]
        axis = 0 if dim == 1 else 1
        del shape[axis]

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
        weight = weight.contiguous()

        # with torch_device_fn.device(inp.device):
        out = unwrap(celoss_kernel[(1,)](
            inp,
            tgt,
            weight,
            reduction,
            ignore_index,
            label_smoothing,))

        if inp.requires_grad:
            ctx.save_for_backward(inp, tgt, weight)
            ctx.ignore_index = ignore_index
            ctx.label_smoothing = label_smoothing
            ctx.shape = shape
            ctx.reduction = reduction

        return out.to(inp.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS CrossEntropyLoss VJP")

        inp, tgt, weight = ctx.saved_tensors

        ignore_index = ctx.ignore_index
        label_smoothing = ctx.label_smoothing
        shape = ctx.shape
        reduction = ctx.reduction
        
        out_grad = out_grad.contiguous()
        inp_grad = torch.zeros(inp.shape, dtype=inp.dtype, device=inp.device)
        inp_grad = unwrap(celoss_bwd_kernel[(1,)](
                out_grad,
                inp,
                tgt,
                weight,
                reduction,
                ignore_index,
                label_smoothing,))
        return inp_grad, None, None, None, None, None


def cross_entropy_loss(
    inp, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    return CrossEntropyLoss.apply(
        inp, target, weight, reduction, ignore_index, label_smoothing
    )
