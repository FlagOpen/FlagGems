import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sin_func(x):
    return tl.sin(x.to(tl.float32))


def sin(A):
    logger.debug("GEMS_CAMBRICON SIN")
    return sin_func(A)


def sin_(A):
    logger.debug("GEMS_CAMBRICON SIN_")
    sin_func(A, out0=A)
    return A
