import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

try:
    import torch_npu  # noqa: F401

    pow = tl.extra.ascend.libdevice.pow
    _tanh = tl.extra.ascend.libdevice.tanh
except:  # noqa: E722
    pow = tl_extra_shim.pow
    _tanh = tl_extra_shim.tanh


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_forward(x):
    return _tanh(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward(y, dy):
    return dy * (1.0 - pow(y.to(tl.float32), 2))


class Tanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        print("GEMS TANH FORWARD")
        if A.requires_grad is True:
            out = tanh_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = tanh_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        print("GEMS TANH BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = tanh_backward(out, out_grad)
        return in_grad


def tanh(A):
    return Tanh.apply(A)


class InplaceTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        print("GEMS TANH_ FORWARD")
        if A.requires_grad is True:
            out = tanh_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            A.copy_(out.to(A.dtype))
            ctx.mark_dirty(A)
        else:
            tanh_forward(A, out0=A)
        return A

    @staticmethod
    def backward(ctx, out_grad):
        print("GEMS TANH_ BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = tanh_backward(out, out_grad)
        return in_grad


def tanh_(A):
    InplaceTanh.apply(A)
    return A
