import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def log_softmax_forward(x, dim : tl.constexpr):
    # y = (x - max) - log(sum(exp(x - max)))

    # (M, N) => (M)
    (values, ) = tl.max_(x, axis=dim)
    # (M, 1)
    values = tl.expand_dims(values, -1)
    # (M, N)
    values = tl.broadcast_to(values, x.shape)
    # (M, N)
    x_tmp = x - values
    # (M, N)
    exp_tmp = tl.exp(x_tmp)
    # (M)
    exp_tmp = tl.sum_(exp_tmp, axis=dim)
    # (M)
    log_tmp = tl.log(exp_tmp)
    # (M, 1)
    log_tmp = tl.expand_dims(log_tmp, -1)
    # (M, N)
    log_tmp = tl.broadcast_to(log_tmp, x.shape)

    # (M, N)
    out = x_tmp - log_tmp

    return out


@triton.jit
def log_softmax_backward(x, out_grad, dim : tl.constexpr):
    # dx = dy - softmax(x) * sum(dy)

    # (M, N)
    tmp_x = x.to(tl.float32)
    softmax_tmp = tl.exp(tmp_x)
    tmp_out_grad = out_grad.to(tl.float32)
    # (M, N) => (M)
    tmp = tl.sum_(tmp_out_grad, axis=dim)
    # (M, 1)
    tmp = tl.expand_dims(tmp, -1)
    # (M, N)
    tmp = tl.broadcast_to(tmp, x.shape)

    x_grad = tmp_out_grad - softmax_tmp * tmp

    return x_grad.to(x.type.element_ty)


class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        logging.debug("GEMS LOG_SOFTMAX FORWARD")

        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        inp = x.contiguous()
        out = unwrap(log_softmax_forward[(1,)](inp, dim))
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS LOG_SOFTMAX VJP")
        (out,) = ctx.saved_tensors

        dim = ctx.dim
        assert dim >= -out.ndim and dim < out.ndim, "Invalid dim"

        out_grad = out_grad.contiguous()
        in_grad = unwrap(log_softmax_backward[(1,)](out, out_grad, dim))
        return in_grad, None


def log_softmax(x, dim=-1, dtype=None):
    if dim < 0:
        dim = x.ndim + dim
    if dim >= x.ndim or dim < 0:
        raise ValueError(f"dim {dim} is out of bounds for tensor of dimension {x.ndim}")
    return LogSoftmax.apply(x, dim)
