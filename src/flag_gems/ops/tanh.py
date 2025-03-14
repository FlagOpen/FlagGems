import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def tanh_forward(x):
    out = tl.tanh(x.to(tl.float32))
    return out.to(x.type.element_ty)


@triton.jit
def tanh_backward(y, dy, ONES, TWO):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    out =  dy_f32 * (ONES - tl.pow(y_f32, TWO))
    return out.to(y.type.element_ty)


class Tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS TANH FORWARD")
        if A.requires_grad is True:
            out = unwrap(tanh_forward[(1,)](A))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = unwrap(tanh_forward[(1,)](A))
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS TANH BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = unwrap(tanh_backward[(1,)](out, out_grad, 1.0, 2.0))
        return in_grad


def tanh(A):
    return Tanh.apply(A)
