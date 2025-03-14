import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit()
def dropout_forward_kernel(inp, p:tl.constexpr, train:tl.constexpr):
    return tl.dropout(inp, p, train)


@triton.jit()
def dropout_backward_kernel(inp, mask, scale : tl.constexpr):
    return tl.dropout_backward(inp, mask, scale)


class NativeDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, p, train):
        logging.debug("GEMS NATIVE DROPOUT FORWARD")
        assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
        inp = inp.contiguous()
        out = unwrap(dropout_forward_kernel[(1,)](inp, p, train))
        ctx.p = p
        return out, None

    @staticmethod
    def backward(ctx, grad_outputs, kwargs):
        logging.debug("GEMS NATIVE DROPOUT BACKWARD")
        (mask, ) = ctx.saved_tensors
        scale = 1.0 / (1.0 - ctx.p)
        grad_outputs = grad_outputs.contiguous()
        grad_inputs = torch.empty_like(grad_outputs)
        grad_inputs = unwrap(dropout_backward_kernel[(1,)](grad_outputs, mask, scale))
        return grad_inputs, None, None


def native_dropout(x, p=0.5, train=True):
    return NativeDropout.apply(x, p, train)

