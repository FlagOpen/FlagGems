import logging

import torch
import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def group_norm_kernel(
    input,
    W,
    B,
    N : tl.constexpr,
    C : tl.constexpr,
    HW : tl.constexpr,
    num_groups : tl.constexpr,
    eps : tl.constexpr,
    outdtype : tl.constexpr,
    axis_dim : tl.constexpr
):
    input = input.to(tl.float32)
    W = W.to(tl.float32)
    B = B.to(tl.float32)

    # compute mean
    # (N, num_groups, C * HW // num_groups)
    reshaped_input = tl.reshape(input, (N, num_groups, C * HW // num_groups))
    # (N, num_groups)
    mean = tl.sum_(reshaped_input, axis=2) / reshaped_input.shape[2]

    # compute var and rstd
    # (N, num_groups, 1)
    mean_tmp = tl.expand_dims(mean, -1)
    # (N, num_groups, C * HW // num_groups)
    mean_tmp = tl.broadcast_to(mean_tmp, (N, num_groups, C * HW // num_groups))
    # (N, num_groups, C * HW // num_groups)
    tmp = reshaped_input - mean_tmp
    # (N, num_groups, C * HW // num_groups)
    var = tmp * tmp
    # (N, num_groups, 1)
    var = tl.sum_(var, axis=2) / reshaped_input.shape[2]
    # (N, num_groups)
    rstd = tl.rsqrt(var + eps)
    # (N, num_groups, 1)
    rstd_tmp = tl.expand_dims(rstd, -1)
    # (N, num_groups, C * HW // num_groups)
    rstd_tmp = tl.broadcast_to(rstd_tmp, tmp.shape)

    # compute output
    output = tmp * rstd_tmp
    # (N, C, H, W)
    output = tl.reshape(output, input.shape)
    # (1, C, 1, 1)
    W = tl.expand_dims(W, axis_dim)
    W = tl.broadcast_to(W, output.shape)
    # (1, C, 1, 1)
    B = tl.expand_dims(B, axis_dim)
    B = tl.broadcast_to(B, output.shape)
    output = output * W + B
    output = output.to(outdtype)
    mean = mean.to(outdtype)
    rstd = rstd.to(outdtype)
    return output, mean, rstd


@triton.jit
def group_norm_backward_kernel(
    y_grad,
    mean,
    rstd,
    x,
    num_groups : tl.constexpr,
    weight,
    N : tl.constexpr,
    C : tl.constexpr,
    HW : tl.constexpr,
    outdtype : tl.constexpr,
    axis_dim : tl.constexpr
):
    y_grad = y_grad.to(tl.float32)
    mean = mean.to(tl.float32)
    rstd = rstd.to(tl.float32)
    x = x.to(tl.float32)
    weight = weight.to(tl.float32)

    M : tl.constexpr = C * HW // num_groups

    # y = weight * x_norm + bias
    # dbias = sum(dy)
    # (N, C, H, W) ==> (C)
    bias_grad = tl.sum_(y_grad, axis=axis_dim)

    # (N, num_groups) ==> (N, num_groups, 1)
    mean_reshape = tl.expand_dims(mean, -1)
    # (N, num_groups, 1) ==> (N, num_groups, C * HW // num_groups)
    mean_reshape = tl.broadcast_to(mean_reshape, (N, num_groups, M))

    # (N, num_groups) ==> (N, num_groups, 1)
    rstd_reshape = tl.expand_dims(rstd, -1)
    # (N, num_groups, 1) ==> (N, num_groups, C * HW // num_groups)
    rstd_reshape = tl.broadcast_to(rstd_reshape, (N, num_groups, M))

    # (N, C, H, W)    ==> (N, num_groups, C * HW // num_groups)
    x_reshape = tl.reshape(x, (N, num_groups, M))

    # y = weight * x_norm + bias
    # x_norm = (x - mean) * rstd
    # dweight = sum(dy * x_norm)
    tmp = x_reshape - mean_reshape
    x_norm = tmp * rstd_reshape
    # (N, num_groups, C * HW // num_groups) ==> (N, C, H, W)
    x_norm_reshape = tl.reshape(x_norm, x.shape)
    # (N, C, H, W) ==> (C)
    weight_grad = tl.sum_(y_grad * x_norm_reshape, axis=axis_dim)

    # dx = dx_norm * rstd + dvar * 2.0 * (x - mean) / M + dmean / M

    # dx_norm = dy * weight
    # (C, 1) ==> (1, C, 1, 1)
    weight_reshape = tl.expand_dims(weight, axis_dim)
    # (1, C, 1, 1) ==> (N, C, H, W)
    weight_reshape = tl.broadcast_to(weight_reshape, x.shape)
    # (N, C, H, W)
    dx_norm = weight_reshape * y_grad
    # (N, C, H, W) ==> (N, num_groups, C * HW // num_groups)
    dx_norm_reshape = tl.reshape(dx_norm, (N, num_groups, M))

    # dvar = sum (dx_norm * (-0.5) * (x - mean) * rstd ** 3)
    # (N, num_groups, C * HW // num_groups)
    dvar_tmp = (-0.5) * x_norm * rstd_reshape * rstd_reshape * dx_norm_reshape
    # (N, num_groups)
    dvar = tl.sum_(dvar_tmp, axis=2)
    # (N, num_groups, 1)
    dvar_reshape = tl.expand_dims(dvar * 2.0 / M, -1)
    # (N, num_groups, C * HW // num_groups)
    dvar_reshape = tl.broadcast_to(dvar_reshape, (N, num_groups, M))

    # dmean = sum (dx_norm * (-1.0) * rstd)
    #       + dvar * sum ((-2.0) * (x - mean) / (C * HW // num_groups))
    # (N, num_groups, C * HW // num_groups) ==> (N, num_groups)
    dmean_tmp1 = tl.sum_(dx_norm_reshape * (-1.0) * rstd_reshape, axis=2)
    # (N, num_groups, C * HW // num_groups) ==> (N, num_groups)
    dmean_tmp2 = tl.sum_((-2.0) * tmp / M, axis=2)
    # (N, num_groups)
    dmean = dmean_tmp1 + dvar * dmean_tmp2
    # (N, num_groups, 1)
    dmean_reshape = tl.expand_dims(dmean / M, -1)
    # (N, num_groups, C * HW // num_groups)
    dmean_reshape = tl.broadcast_to(dmean_reshape, (N, num_groups, M))

    # (N, num_groups, C * HW // num_groups)
    dx = dx_norm_reshape * rstd_reshape + dvar_reshape * tmp + dmean_reshape
    # (N, C, H, W)
    x_grad = tl.reshape(dx, x.shape)

    x_grad = x_grad.to(outdtype)
    weight_grad = weight_grad.to(outdtype)
    bias_grad = bias_grad.to(outdtype)

    return x_grad, weight_grad, bias_grad

def type_convert(dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.bfloat16:
        return tl.bfloat16

class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, N, C, HW, num_groups, weight=None, bias=None, eps=1e-05):
        logging.debug("GEMS GROUPNORM FORWARD")
        group_size = C // num_groups
        x = x.contiguous()
        if weight is not None:
            weight = weight.contiguous()
        else:
            weight = torch.ones((C,), dtype=x.dtype, device=x.device)
        if bias is not None:
            bias = bias.contiguous()
        else:
            bias = torch.zeros((C,), dtype=x.dtype, device=x.device)
        y = torch.empty_like(x)
        mean = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)
        rstd = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)

        axis_dim = [i for i in range(len(x.shape)) if i != 1]

        (y, mean, rstd) = unwrap(group_norm_kernel[(1,)](
            x,
            weight,
            bias,
            N,
            C,
            HW,
            num_groups,
            eps,
            type_convert(x.dtype),
            axis_dim
        ))
        if x.requires_grad:
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            ctx.num_groups = num_groups
            ctx.group_size = group_size
            ctx.N = N
            ctx.C = C
            ctx.HW = HW
        return y, mean, rstd

    @staticmethod
    def backward(ctx, y_grad, mean_grad, rstd_grad):
        logging.debug("GEMS GROUPNORM BACKWARD")
        y_grad = y_grad.contiguous()
        (x, weight, bias, mean, rstd) = ctx.saved_tensors
        num_groups = ctx.num_groups
        N = ctx.N
        C = ctx.C
        HW = ctx.HW
        x_grad = torch.empty_like(x)
        weight_grad = torch.empty_like(weight)
        bias_grad = torch.empty_like(bias)

        axis_dim = [i for i in range(len(y_grad.shape)) if i != 1]

        (x_grad, weight_grad, bias_grad) = unwrap(group_norm_backward_kernel[(1,)](
            y_grad,
            mean,
            rstd,
            x,
            num_groups,
            weight,
            N,
            C,
            HW,
            type_convert(x.dtype),
            axis_dim
        ))
        return (
            x_grad,      # grad for x
            None,        # grad for N (integer, doesn't need gradient)
            None,        # grad for C (integer, doesn't need gradient)
            None,        # grad for HW (integer, doesn't need gradient)
            None,        # grad for num_groups (integer, doesn't need gradient)
            weight_grad, # grad for weight
            bias_grad,   # grad for bias
            None         # grad for eps (float, doesn't need gradient)
        )


def group_norm(x, weight, bias, N, C, HW, num_groups, eps):
    return GroupNorm.apply(x, N, C, HW, num_groups, weight, bias, eps)
