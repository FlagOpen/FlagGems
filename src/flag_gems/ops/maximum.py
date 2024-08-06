import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def maximum_kernel(X, Y):
    return tl.maximum(X, Y)


def maximum(X, Y):
    logging.debug("GEMS MAXIMUM")
    assert X.is_cuda and Y.is_cuda
    assert X.ndim == 1 and Y.ndim == 1 and X.size(0) == Y.size(0)
    return maximum_kernel(X, Y)
