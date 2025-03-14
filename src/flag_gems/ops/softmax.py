import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def softmax_forward_func(x, dim: tl.constexpr):
    return tl.softmax(x, dim)

@triton.jit
def softmax_backward_func(x, grad, dim: tl.constexpr):
    return tl.softmaxbwd(x, grad, dim)

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        logging.debug("GEMS SOFTMAX")

        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim
        inp = x.contiguous()
        out = unwrap(softmax_forward_func[(1,)](inp, dim))

        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS SOFTMAX VJP")
        dim = ctx.dim
        (out,) = ctx.saved_tensors

        assert dim >= -out.ndim and dim < out.ndim, "Invalid dim"
        dim = dim % out.ndim

        out_grad = out_grad.contiguous()
        in_grad = unwrap(softmax_backward_func[(1,)](out, out_grad, dim))
        return in_grad, None, None


def softmax(x, dim=-1, dtype=None):
    return Softmax.apply(x, dim, dtype)

