import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def relu_forward(x):
    out = tl.maximum(0, x)
    return out.to(x.type.element_ty)


@triton.jit
def relu_backward(x, dy, ZEROS):
    a = x.to(tl.float32) > ZEROS
    return a * dy


class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS RELU FORWARD")
        out = unwrap(relu_forward[(1,)](A))
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS RELU BACKWARD")
        (inp,) = ctx.saved_tensors
        in_grad = unwrap(relu_backward[(1,)](inp, out_grad, 0.0))
        return in_grad


def relu(A):
    return Relu.apply(A)
