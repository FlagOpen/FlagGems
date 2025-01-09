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
