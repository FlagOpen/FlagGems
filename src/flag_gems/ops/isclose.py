import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic
from .all import all


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func(x, y, rtol, atol):
    x_fp = x.to(tl.float64)
    y_fp = y.to(tl.float64)
    return tl.where(
        tl.math.isinf(x.to(tl.float32)) | tl.math.isinf(y.to(tl.float32)),
        x_fp == y_fp,
        tl.abs(x_fp - y_fp) <= atol + rtol * tl.abs(y_fp),
    )


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func_equal_nan(x, y, rtol, atol):
    x_fp = x.to(tl.float64)
    y_fp = y.to(tl.float64)
    x_nan = x_fp != x_fp
    y_nan = y_fp != y_fp
    return tl.where(
        x_nan | y_nan,
        x_nan == y_nan,
        tl.where(
            tl.math.isinf(x.to(tl.float32)) | tl.math.isinf(y.to(tl.float32)),
            x_fp == y_fp,
            tl.abs(x_fp - y_fp) <= atol + rtol * tl.abs(y_fp),
        ),
    )


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func_int(x, y, rtol, atol):
    return tl.abs(x - y) <= atol + rtol * tl.abs(y)


def isclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> torch.Tensor:
    logging.debug("GEMS ISCLOSE")
    if rtol < 0:
        raise RuntimeError(
            "rtol must be greater than or equal to zero, but got {}".format(rtol)
        )
    if atol < 0:
        raise RuntimeError(
            "atol must be greater than or equal to zero, but got {}".format(atol)
        )
    def is_int(X):
        return X.dtype == torch.int8 or X.dtype == torch.int16 or \
            X.dtype == torch.int32 or X.dtype == torch.int64
    if False and is_int(A) and is_int(B):
        return isclose_func_int(A, B, rtol, atol)
    elif equal_nan:
        return isclose_func_equal_nan(A, B, rtol, atol)
    else:
        return isclose_func(A, B, rtol, atol)


def allclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> bool:
    logging.debug("GEMS ALLCLOSE")
    return all(isclose(A, B, rtol, atol, equal_nan)).item()
