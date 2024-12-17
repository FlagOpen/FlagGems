import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def logical_xor_func(x, y):
    return x.to(tl.int1) ^ y.to(tl.int1)


def logical_xor(A, B):
    logging.debug("GEMS LOGICAL_XOR")
    return logical_xor_func(A, B)
