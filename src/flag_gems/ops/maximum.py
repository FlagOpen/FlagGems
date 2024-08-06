import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 0, "DEFAULT")])
@triton.jit
def maximum_kernel(X, Y):
    return tl.maximum(X, Y)


def maximum(X, Y):
    logging.debug("GEMS MAXIMUM")
    assert X.is_cuda and Y.is_cuda
    return maximum_kernel(X, Y)
