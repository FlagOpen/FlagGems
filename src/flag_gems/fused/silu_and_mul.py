import torch
import triton
import triton.language as tl
import logging
from ..utils import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return x_silu * y


class SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        logging.debug("GEMS SILU AND MUL FORWARD")
        O = silu_and_mul_kernel(A, B)
        return O


def silu_and_mul(A, B):
    return SiluAndMul.apply(A, B)
