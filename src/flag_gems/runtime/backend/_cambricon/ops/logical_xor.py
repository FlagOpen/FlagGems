import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def logical_xor_func(x, y):
    return x.to(tl.int1) ^ y.to(tl.int1)


def logical_xor(A, B):
    logger.debug("GEMS_CAMBRICON LOGICAL_XOR")
    return logical_xor_func(A, B)
