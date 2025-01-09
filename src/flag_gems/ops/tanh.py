import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

pow = tl_extra_shim.pow
_tanh = tl_extra_shim.tanh


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_kernel(x):
    return _tanh(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward_kernel(y, dy):
    return dy * (1.0 - y * y)


def tanh(self):
    logging.debug("GEMS TANH FORWARD")
    out = tanh_kernel(self)
    return out


def tanh_backward(grad_output, output):
    logging.debug("GEMS TANH BACKWARD")
    in_grad = tanh_backward_kernel(output, grad_output)
    return in_grad


class InplaceTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS TANH_ FORWARD")
        if A.requires_grad is True:
            out = tanh_kernel(A.to(torch.float32))
            ctx.save_for_backward(out)
            A.copy_(out.to(A.dtype))
            ctx.mark_dirty(A)
        else:
            tanh_kernel(A, out0=A)
        return A

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS TANH_ BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = tanh_backward_kernel(out, out_grad)
        return in_grad


def tanh_(A):
    InplaceTanh.apply(A)
    return A
