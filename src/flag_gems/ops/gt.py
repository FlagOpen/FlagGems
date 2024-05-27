import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic(output_dtypes=[torch.bool])
@triton.jit
def gt_func(x, y):
    return x.to(tl.float32) > y


def gt(A, B):
    logging.debug("GEMS GT")
    O = gt_func(A, B)
    return O


@pointwise_dynamic(is_tensor=[True, False], output_dtypes=[torch.bool])
@triton.jit
def gt_func_scalar(x, y):
    return x.to(tl.float32) > y


def gt_scalar(A, B):
    logging.debug("GEMS GT SCALAR")
    O = gt_func_scalar(A, B)
    return O
