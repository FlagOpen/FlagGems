import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def sigmoid_forward(x):
    return tl.sigmoid_(x)


@triton.jit
def sigmoid_backward(y, dy, ONES):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (ONES - y_f32) * y_f32


class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS SIGMOID FORWARD")
        if A.requires_grad is True:
            out = unwrap(sigmoid_forward[(1,)](A.to(torch.float32)))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = unwrap(sigmoid_forward[(1,)](A))
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS SIGMOID BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = unwrap(sigmoid_backward[(1,)](out, out_grad, 1.0))
        return in_grad


def sigmoid(A):
    return Sigmoid.apply(A)
