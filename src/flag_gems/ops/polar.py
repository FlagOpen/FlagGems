import logging
import triton
import torch
import triton.language as tl
from ..utils import pointwise_dynamic
@pointwise_dynamic(promotion_methods=[(0, "DEFAULT"),  (1, "DEFAULT")])
@triton.jit
def polar_kernel(abs, angle):
    real_part = abs * tl.cos(angle)
    imag_part = abs * tl.sin(angle)
    return real_part, imag_part

def polar(abs, angle):
    logging.debug("GEMS POLAR")
    real, imag = polar_kernel(abs, angle)
    return torch.complex(real, imag)