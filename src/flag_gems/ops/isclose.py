import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic
from .all import all


@pointwise_dynamic(
    is_tensor=[True, True, False, False, False, False, False],
    output_dtypes=[torch.bool],
)
@triton.jit
def isclose_func(
    x,
    y,
    rtol,
    atol,
    float_dtype: tl.constexpr,
    zero_tol: tl.constexpr,
    equal_nan: tl.constexpr,
):
    if zero_tol:
        if equal_nan:
            x_nan = x != x
            y_nan = y != y
            return tl.where(
                x_nan | y_nan,
                x_nan == y_nan,
                x == y,
            )
        else:
            return x == y
    if float_dtype:
        if x.dtype == torch.float64:
            x_fp = x
            y_fp = y
        else:
            x_fp = x.to(tl.float32)
            y_fp = y.to(tl.float32)
        if equal_nan:
            x_nan = x_fp != x_fp
            y_nan = y_fp != y_fp
            return tl.where(
                x_nan | y_nan,
                x_nan == y_nan,
                tl.where(
                    tl.math.isinf(x_fp) | tl.math.isinf(y_fp),
                    x_fp == y_fp,
                    tl.abs(x - y) <= atol + rtol * tl.abs(y),
                ),
            )
        else:
            return tl.where(
                tl.math.isinf(x_fp) | tl.math.isinf(y_fp),
                x_fp == y_fp,
                tl.abs(x - y) <= atol + rtol * tl.abs(y),
            )
    else:
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
    float_dtype = A.dtype in (
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
    )
    zero_atol = rtol == 0 and atol == 0
    return isclose_func(A, B, rtol, atol, float_dtype, zero_atol, equal_nan)


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
