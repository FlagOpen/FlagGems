import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(output_dtypes=[torch.bool])
@triton.jit
def isinf_func(x):
    return tl.extra.mlu.libdevice.isinf(x.to(tl.float32))


def isinf(A):
    logging.debug("GEMS ISINF")
    return isinf_func(A)
