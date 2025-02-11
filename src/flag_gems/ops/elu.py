import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def elu_forward_kernel(x, alpha):
    return tl.where(x > 0, x, alpha * (tl.exp(x.to(tl.float32)) - 1))


def elu(A, alpha=1.0):
    logging.debug("GEMS ELU")
    return elu_forward_kernel(A, alpha)
