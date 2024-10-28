import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic

try:
    from triton.language.extra.cuda.libdevice import pow
except ImportError:
    try:
        from triton.language.math import pow
    except ImportError:
        from triton.language.libdevice import pow

try:
    from triton.language.extra.cuda.libdevice import tanh as _tanh
except ImportError:
    try:
        from triton.language.math import tanh as _tanh
    except ImportError:
        from triton.language.libdevice import tanh as _tanh

"""
#------------- this is a torch2.4 solution ---------------#

@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_forward_kernel(x: torch.Tensor):
    return _tanh(x.to(tl.float32))

@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward_kernel(y: torch.Tensor, dy: torch.Tensor):
    return dy * (1.0 - pow(y.to(tl.float32), 2))

@torch.library.custom_op("gems::tanh_forward", mutates_args=())
def tanh_forward(x: torch.Tensor) -> torch.Tensor:
    return tanh_forward_kernel(x)

@torch.library.custom_op("gems::tanh_backward", mutates_args=())
def tanh_backward(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    return tanh_backward_kernel(y, dy)

@torch.library.impl_abstract("gems::tanh_forward")
def fake_tanh_forward(x: torch.Tensor) -> torch.Tensor:
    return x

@torch.library.impl_abstract("gems::tanh_backward")
def fake_tanh_backward(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    return dy
"""

torch.library.define("gems::tanh_forward", "(Tensor x) -> Tensor")
torch.library.define("gems::tanh_backward", "(Tensor y, Tensor dy) -> Tensor")

@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_forward_kernel(x: torch.Tensor):
    return _tanh(x.to(tl.float32))

@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward_kernel(y: torch.Tensor, dy: torch.Tensor):
    return dy * (1.0 - pow(y.to(tl.float32), 2))

@torch.library.impl("gems::tanh_forward", "cuda")
def tanh_forward(x: torch.Tensor) -> torch.Tensor:
    return tanh_forward_kernel(x)

@torch.library.impl("gems::tanh_backward", "cuda")
def tanh_backward(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    return tanh_backward_kernel(y, dy)

@torch.library.impl_abstract("gems::tanh_forward")
def fake_tanh_forward(x: torch.Tensor) -> torch.Tensor:
    return x

@torch.library.impl_abstract("gems::tanh_backward")
def fake_tanh_backward(y: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    return dy


class Tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("GEMS TANH FORWARD")
        if A.requires_grad is True:
            out = torch.ops.gems.tanh_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = torch.ops.gems.tanh_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS TANH BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = torch.ops.gems.tanh_backward(out, out_grad)
        return in_grad

def tanh(A: torch.Tensor):
    return Tanh.apply(A)
    