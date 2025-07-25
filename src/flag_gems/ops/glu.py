import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
exp = tl_extra_shim.exp


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def glu_kernel(a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    result = a * sigmoid_b

    return result


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")] )
@triton.jit
def glu_backward_kernel(grad_output, a, b):
    sigmoid_b = 1 / (1 + exp(-b.to(tl.float32)))
    da = grad_output * sigmoid_b
    s_b = sigmoid_b.to(a.dtype)
    db = grad_output * a * s_b * (1 - s_b)

    return da, db


def glu(self, dim=-1):
    assert self.shape[dim] % 2 == 0, "Split dimension must be even"
    logger.debug("GLU FORWARD")
    # Split into a and b
    a, b = torch.chunk(self, 2, dim=dim)
    out = glu_kernel(a, b)

    return out


def glu_backward(grad_output, self, dim=-1):
    assert self.shape[dim] % 2 != 0, "Split dimension must be even for GLU backward"
    logger.debug("GEMS GLU BACKWARD")  
    # Recreate a and b
    a, b = torch.chunk(self, 2, dim=dim)
    grad_a, grad_b = glu_backward_kernel(grad_output.contiguous(), a.contiguous(), b.contiguous())
    grad_input = torch.cat([grad_a, grad_b], dim=dim)
    
    return grad_input