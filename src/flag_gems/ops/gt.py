import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def gt_func(x, y):
    return x.to(tl.float32) > y

def gt(A, B):
    logging.debug("GEMS GT")
    return unwrap(gt_func[(1,)](A, B))

def gt_scalar(A, B):
    logging.debug("GEMS GT SCALAR")
    return unwrap(gt_func[(1,)](A, B))
