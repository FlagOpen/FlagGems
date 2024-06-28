import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic
from .all import all


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func(x, y, rtol, atol):
    x_fp = x.to(tl.float32)
    y_fp = y.to(tl.float32)
    return tl.where(
        tl.math.isinf(x_fp) | tl.math.isinf(y_fp),
        x_fp == y_fp,
        tl.abs(x_fp - y_fp) <= atol + rtol * tl.abs(y_fp),
    )


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func_equal_nan(x, y, rtol, atol):
    x_fp = x.to(tl.float32)
    y_fp = y.to(tl.float32)
    x_nan = x_fp != x_fp
    y_nan = y_fp != y_fp
    return tl.where(
        x_nan | y_nan,
        x_nan == y_nan,
        tl.where(
            tl.math.isinf(x_fp) | tl.math.isinf(y_fp),
            x_fp == y_fp,
            tl.abs(x_fp - y_fp) <= atol + rtol * tl.abs(y_fp),
        ),
    )


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func_fp(x, y, rtol, atol):
    return tl.where(
        tl.math.isinf(x) | tl.math.isinf(y),
        x == y,
        tl.abs(x - y) <= atol + rtol * tl.abs(y),
    )


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func_equal_nan_fp(x, y, rtol, atol):
    x_nan = x != x
    y_nan = y != y
    return tl.where(
        x_nan | y_nan,
        x_nan == y_nan,
        tl.where(
            tl.math.isinf(x) | tl.math.isinf(y),
            x == y,
            tl.abs(x - y) <= atol + rtol * tl.abs(y),
        ),
    )


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func_int(x, y, rtol, atol):
    x_long = x.to(tl.int64)
    y_long = y.to(tl.int64)
    return tl.abs(x_long - y_long) <= atol + rtol * tl.abs(y_long)


def _isclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> torch.Tensor:
    if A.dtype != B.dtype:
        raise RuntimeError("{} did not match {}".format(A.dtype, B.dtype))
    if A.is_quantized or B.is_quantized:
        raise RuntimeError("isclose is not supported for quantized inputs.")
    if rtol < 0:
        raise RuntimeError(
            "rtol must be greater than or equal to zero, but got {}".format(rtol)
        )
    if atol < 0:
        raise RuntimeError(
            "atol must be greater than or equal to zero, but got {}".format(atol)
        )

    if (
        A.dtype == torch.int64
        or A.dtype == torch.int32
        or A.dtype == torch.int16
        or A.dtype == torch.int8
        or A.dtype == torch.bool
    ):
        return isclose_func_int(A, B, rtol, atol)
    elif equal_nan:
        if A.dtype == torch.float32 or A.dtype == torch.float64:
            return isclose_func_equal_nan_fp(A, B, rtol, atol)
        else:
            return isclose_func_equal_nan(A, B, rtol, atol)
    else:
        if A.dtype == torch.float32 or A.dtype == torch.float64:
            return isclose_func_fp(A, B, rtol, atol)
        else:
            return isclose_func(A, B, rtol, atol)


def isclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> torch.Tensor:
    logging.debug("GEMS ISCLOSE")
    return _isclose(A, B, rtol, atol, equal_nan)


def allclose(
    A: torch.Tensor,
    B: torch.Tensor,
    rtol=1e-05,
    atol=1e-08,
    equal_nan: bool = False,
) -> bool:
    logging.debug("GEMS ALLCLOSE")
    return all(_isclose(A, B, rtol, atol, equal_nan)).item()
