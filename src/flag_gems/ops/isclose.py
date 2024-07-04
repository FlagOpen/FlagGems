import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic
from .all import all


@pointwise_dynamic(
    is_tensor=[True, True, False, False, False, False, False, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
)
@triton.jit
def isclose_func(
    x,
    y,
    rtol,
    atol,
    float_dtype: tl.constexpr,
    reduced_dtype: tl.constexpr,
    zero_tol: tl.constexpr,
    equal_nan: tl.constexpr,
):
    cast_x = x if x.dtype == tl.float64 else x.to(tl.float32)
    cast_y = y if x.dtype == tl.float64 else y.to(tl.float32)
    if x.dtype == tl.bfloat16:
        close = cast_x == cast_y
    elif reduced_dtype:
        close = cast_x == cast_y
    else:
        close = x == y
    if equal_nan:
        if float_dtype:
            close |= (cast_x != cast_x) & (cast_y != cast_y)
    if zero_tol:
        return close
    else:
        allowed_error = atol + tl.abs(rtol * cast_y)
        actual_error = tl.abs(cast_x - cast_y)
        actual_error_finite = (actual_error != float("inf")) & (
            actual_error != float("-inf")
        )
        return close | (actual_error_finite & (actual_error <= allowed_error))


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
    reduced_dtype = A.dtype in (torch.bool, torch.int8)
    zero_tol = (rtol == 0) and (atol == 0)
    return isclose_func(
        A, B, rtol, atol, float_dtype, reduced_dtype, zero_tol, equal_nan
    )


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
