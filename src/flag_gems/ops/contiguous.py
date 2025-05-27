import logging

import torch
import triton

from ..ops.copy import copy
from ..utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def contiguous_backward(out, grad):
    return grad


ALL_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


class Contiguous(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        logging.debug("GEMS CONTIGUOUS FORWARD")
        if inp.is_contiguous(memory_format=torch.contiguous_format):
            out = inp
        else:
            if inp.dtype in ALL_FLOAT_DTYPES:
                output = torch.empty_like(inp, memory_format=torch.contiguous_format)
            else:
                output = torch.empty_like(inp, memory_format=torch.contiguous_format)
            out = copy(inp, out0=output)
        if inp.requires_grad is True:
            ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS CONTIGUOUS BACKWARD")
        out_grad = out_grad.contiguous()
        (out,) = ctx.saved_tensors
        if out.shape == out_grad.shape:
            return out_grad
        out_grad = contiguous_backward(out, out_grad, out0=out)
        return out_grad


def contiguous(inp, memory_format=torch.contiguous_format):
    assert memory_format == torch.contiguous_format
    return Contiguous.apply(inp)
