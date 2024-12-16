import logging

import triton
import triton.language as tl

from ..runtime import device
from ..utils import pointwise_dynamic

device = device.name


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def maximum_kernel(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)

    return tl.maximum(X, Y)


def maximum(X, Y):
    logging.debug("GEMS MAXIMUM")
    assert X.device.type == device and Y.device.type == device
    return maximum_kernel(X, Y)
