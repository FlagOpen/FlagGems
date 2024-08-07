import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(
    is_tensor=[True, True], dtypes=[float, float], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def maximum_kernel(X, Y):
    return tl.maximum(X, Y)


def convert_bf16_to_fp32(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.to(torch.float32)
    return tensor


def maximum(X, Y):
    logging.debug("GEMS MAXIMUM")
    assert X.is_cuda and Y.is_cuda
    X = convert_bf16_to_fp32(X)
    Y = convert_bf16_to_fp32(Y)
    return maximum_kernel(X, Y)
