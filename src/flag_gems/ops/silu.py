import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap

@triton.jit
def silu_forward(x):
    return tl.silu(x)

@triton.jit
def silu_backward(x, dy):
    return tl.silubwd(x, dy)

class Silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS SILU FORWARD")
        out = unwrap(silu_forward[(1,)](A))
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS SILU BACKWARD")
        (inp,) = ctx.saved_tensors
        in_grad = unwrap(silu_backward[(1,)](inp, out_grad))
        return in_grad

def silu(A):
    return Silu.apply(A)

