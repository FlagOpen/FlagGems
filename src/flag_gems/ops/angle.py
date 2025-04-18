import triton
import triton.language as tl
import torch
import math
from ..utils import pointwise_dynamic,tl_extra_shim

atan2 = tl_extra_shim.atan2

@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def angle_func(real, imag):
    pi = math.pi
    zero = 0.0
    one = 1.0

    real_zero_mask = real == zero
    imag_positive = imag > zero


    angle_if_real_zero = tl.where(
        imag_positive, pi / 2,
        tl.where(imag < zero, -pi / 2, zero)
    )
    real_last=tl.where(real.dtype==torch.float16,real/one,real)
    imag_last=tl.where(imag.dtype==torch.float16,imag/one,imag)
    atan_x = atan2(imag_last,real_last)
    return tl.where(real_zero_mask, angle_if_real_zero, atan_x)

@pointwise_dynamic(
    is_tensor=[True],
    promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def angle_float(real):
        zero = 0.0
        pi = math.pi
        real_positive= real>=zero
        result=tl.where(real_positive,zero,pi)
        return result

def angle(input_tensor: torch.Tensor) -> torch.Tensor:
    if input_tensor.dtype == torch.complex32:
       real = input_tensor.real
       imag = input_tensor.imag
       result = angle_func(real, imag)
       return result.to(torch.float16)
    elif input_tensor.dtype == torch.complex64:
       real = input_tensor.real
       imag = input_tensor.imag
       result = angle_func(real, imag)
       return result.to(torch.float32)
    elif input_tensor.dtype == torch.float:
       real = input_tensor
       return angle_float(real)
