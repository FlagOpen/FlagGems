import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 0, "DEFAULT")])
@triton.jit
def minimum_kernel(X, Y):
    return tl.minimum(X, Y)


def convert_bf16_to_fp32(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.to(torch.float32)
    return tensor


def minimum(X, Y):
    logging.debug("GEMS MINIMUM")
    assert X.is_cuda and Y.is_cuda
    X = convert_bf16_to_fp32(X)
    Y = convert_bf16_to_fp32(Y)
    return minimum_kernel(X, Y)
