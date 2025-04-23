import math

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

atan2 = tl_extra_shim.atan2


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def angle_func(real, imag):
    one = 1.0
    real_last = tl.where(real.dtype == torch.float16, real / one, real)
    imag_last = tl.where(imag.dtype == torch.float16, imag / one, imag)
    atan_x = atan2(imag_last, real_last)
    result = tl.where(real.dtype == torch.float16, atan_x.to(tl.float16), atan_x)
    return result


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def angle_float(real):
    zero = 0.0
    pi = math.pi
    real_positive = real >= zero
    result = tl.where(real_positive, zero, pi)
    return result


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def angle_int(real):
    zero = 0.0
    pi = math.pi
    real_positive = real >= zero
    result = tl.where(real_positive, zero, pi)
    return result


def angle(input_tensor: torch.Tensor) -> torch.Tensor:
    if (
        input_tensor.dtype == torch.int32
        or input_tensor.dtype == torch.int16
        or input_tensor.dtype == torch.int64
        or input_tensor.dtype == torch.bool
    ):
        real = input_tensor
        return angle_int(real)
    elif (
        input_tensor.dtype == torch.float
        or input_tensor.dtype == torch.float16
        or input_tensor.dtype == torch.bfloat16
    ):
        real = input_tensor
        return angle_float(real)
    elif input_tensor.dtype == torch.complex32 or input_tensor.dtype == torch.complex64:
        real = input_tensor.real
        imag = input_tensor.imag
        return angle_func(real, imag)
