import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True], output_dtypes=[torch.bool])
@triton.jit
def isfinite_func(x):
    x_fp = x.to(tl.float32)
    return (x_fp == x_fp) & (x_fp != float("inf")) & (x_fp != float("-inf"))


@pointwise_dynamic(is_tensor=[True], output_dtypes=[torch.bool])
@triton.jit
def isfinite_func_fp(x):
    return (x == x) & (x != float("inf")) & (x != float("-inf"))


def _isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    if (
        A.dtype == torch.int64
        or A.dtype == torch.int32
        or A.dtype == torch.int16
        or A.dtype == torch.int8
        or A.dtype == torch.bool
    ):
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
    else:
        if A.dtype == torch.float32 or A.dtype == torch.float64:
            return isfinite_func_fp(A)
        else:
            return isfinite_func(A)


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logging.debug("GEMS ISFINITE")
    return _isfinite(A)
