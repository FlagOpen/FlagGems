import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

exp2 = tl_extra_shim.exp2


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_forward(x):
    # log2e: tl.constexpr = math.log2(math.e)
    # triton 3.0.0 disallow calling non-jitted function inside jitted function, even if it is in
    # the rhs of an assignment to a constexpr, so we use numeric literal instead to work around this.
    log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + exp2(-x.to(tl.float32) * log2e))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_backward_kernel(dy, y):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


def sigmoid(self):
    logging.debug("GEMS SIGMOID FORWARD")
    output = sigmoid_forward(self)
    return output


def sigmoid_backward(grad_output, output):
    logging.debug("GEMS SIGMOID BACKWARD")
    grad_input = sigmoid_backward_kernel(grad_output, output)
    return grad_input


class InplaceSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS SIGMOID_ FORWARD")
        if A.requires_grad is True:
            out = sigmoid_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            A.copy_(out.to(A.dtype))
        else:
            sigmoid_forward(A, out0=A)
        ctx.mark_dirty(A)
        return A

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS SIGMOID_ BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = sigmoid_backward_kernel(out, out_grad)
        return in_grad


def sigmoid_(A):
    return InplaceSigmoid.apply(A)
