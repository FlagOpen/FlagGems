import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def reciprocal_func(x):
    out = 1.0 / x.to(tl.float32)
    return out.to(x.type.element_ty)


def reciprocal(A):
    logging.debug("GEMS RECIPROCAL")
    return unwrap(reciprocal_func[(1,)](A))
