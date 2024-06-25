import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func(x, y, rtol, atol):
    return tl.abs(x.to(tl.float64) - y.to(tl.float64)) <= atol + rtol * tl.abs(y.to(tl.float64))
    # return tl.abs(x - y) <= atol + rtol * tl.abs(y)


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func_equal_nan(x, y, rtol, atol):
    finite_x = (x == x)
    finite_y = (y == y)
    return tl.where(
        finite_x & finite_y,
        tl.abs(x.to(tl.float64) - y.to(tl.float64)) <= atol + rtol * tl.abs(y.to(tl.float64)),
        finite_x == finite_y
    )


@pointwise_dynamic(is_tensor=[True, True, False, False], output_dtypes=[torch.bool])
@triton.jit
def isclose_func_equal_nan_bf16(x, y, rtol, atol):
    finite_x = (x.to(tl.float64) == x.to(tl.float64))
    finite_y = (y.to(tl.float64) == y.to(tl.float64))
    return tl.where(
        finite_x & finite_y,
        tl.abs(x.to(tl.float64) - y.to(tl.float64)) <= atol + rtol * tl.abs(y.to(tl.float64)),
        finite_x == finite_y
    )


def isclose(
    A : torch.Tensor,
    B : torch.Tensor,
    rtol = 1e-05,
    atol = 1e-08,
    equal_nan : bool = False,
):
    logging.debug("GEMS ISCLOSE")
    if equal_nan:
        if A.dtype == torch.bfloat16 or B.dtype == torch.bfloat16:
            return isclose_func_equal_nan_bf16(A, B, rtol, atol)
        else:
            return isclose_func_equal_nan(A, B, rtol, atol)
    else:
        return isclose_func(A, B, rtol, atol)

