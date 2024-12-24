import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

_isnan = tl_extra_shim.isnan


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isnan_func(x):
    return _isnan(x.to(tl.float32))


def isnan(A):
    logging.debug("GEMS ISNAN")
    return isnan_func(A)
