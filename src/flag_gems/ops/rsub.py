import logging

import torch
import triton

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rsub_func(x, y, alpha):
    return y - x * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def rsub_func_tensor_scalar(x, y, alpha):
    return y - x * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def rsub_func_scalar_tensor(x, y, alpha):
    return y - x * alpha


def rsub(A, B, *, alpha=1):
    logging.debug("GEMS RSUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return rsub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return rsub_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return rsub_func_scalar_tensor(A, B, alpha)
    else:
        # Both scalar
        return B - A * alpha
