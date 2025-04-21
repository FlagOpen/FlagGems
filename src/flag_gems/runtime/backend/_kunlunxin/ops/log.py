import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log_func(x):
    return tl.log(x.to(tl.float32))


def log(A):
    logging.debug("GEMS LOG")
    return log_func(A)
