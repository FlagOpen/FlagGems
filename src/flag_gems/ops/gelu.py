import logging
  
import torch
import triton
import triton.language as tl

from ..utils import unwrap


erf = tl.erf
exp = tl.exp
pow = tl.pow
tanh = tl.tanh

@triton.jit
def gelu_none(x):
    return tl.gelu(x)

@triton.jit
def gelu_tanh(x):
    return tl.gelu(x)

@triton.jit
def gelu_backward_none(x, dy):
    scale1: tl.constexpr = 0.7071067811  # 1 / math.sqrt(2)
    scale2: tl.constexpr = 0.3989422803  # 1 / math.sqrt(2 * math.pi)
    x_fp32 = x.to(tl.float32)
    dydx = (
        scale2 * x_fp32 * exp(-pow(scale1 * x_fp32, 2))
        + 0.5 * erf(scale1 * x_fp32)
        + 0.5
    )
    dx = dydx * dy
    return dx

@triton.jit
def gelu_backward_tanh(x, dy):
    x_fp32 = x.to(tl.float32)
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * pow(x_fp32, 2)))
    dydx = 0.5 * x * (
        (1 - pow(tanh_out, 2)) * (0.79788456 + 0.1070322243 * pow(x_fp32, 2))
    ) + 0.5 * (1 + tanh_out)
    dx = dydx * dy
    return dx

class Gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, approximate):
        logging.debug("GEMS GELU FORWARD")
        if approximate == "tanh":
            out = unwrap(gelu_tanh[(1,)](A))
        else:
            out = unwrap(gelu_none[(1,)](A))
        ctx.save_for_backward(A)
        ctx.approximate = approximate
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS GELU BACKWARD")
        (inp,) = ctx.saved_tensors
        approximate = ctx.approximate
        if approximate == "tanh":
            in_grad = unwrap(gelu_backward_tanh[(1,)](inp, out_grad))
        else:
            in_grad = unwrap(gelu_backward_none[(1,)](inp, out_grad))
        return in_grad, None


def gelu(A, *, approximate="none"):
    return Gelu.apply(A, approximate)

