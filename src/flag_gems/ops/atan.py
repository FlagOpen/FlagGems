import logging
import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_atan = tl_extra_shim.atan
logger = logging.getLogger(__name__)

@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def atan_kernel(x):
    return _atan(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def atan_backward_kernel(x, dy):
    x = x.to(tl.float32)
    dy = dy.to(tl.float32)
    return dy / (1.0 + x * x)


def atan(A):
    logger.debug("GEMS ATAN")
    out = atan_kernel(A)
    return out

def atan_backward(grad_output, input):
    logger.debug("GEMS ATAN BACKWARD")
    in_grad = atan_backward_kernel(input, grad_output)
    return in_grad

def atan_(A):
    logger.debug("GEMS ATAN_")
    atan_kernel(A, out0=A)
    return A