import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.jit(do_not_specialize=["eps"])
def group_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    C,
    HW,
    num_groups,
    eps,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)

    wb_offset = group * group_size + group_offset
    wb_mask = wb_offset < C
    W_ptr = W + wb_offset
    B_ptr = B + wb_offset

    xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
    xy_mask = wb_offset[:, None] < C and hw_offset[None, :] < HW

    Mean_ptr = Mean + pid
    Rstd_ptr = Rstd + pid

    X_ptr = X + xy_offset
    Y_ptr = Y + xy_offset

    mean_base = 0.0
    for xoffset in range(pid * group_size, pid * group_size + group_size, 1):
        for yoffset in range(0, HW, 1):
            row_mask = xoffset < C * num_groups
            col_mask = yoffset < HW
            rcmask = row_mask and col_mask
            offset = xoffset * HW + yoffset
            X_val = tl.load(X + offset, rcmask)
            mean_base = mean_base + X_val
    mean = mean_base / num_elements

    var_base = 0.0
    for xoffset in range(pid * group_size, pid * group_size + group_size, 1):
        for yoffset in range(0, HW, 1):
            row_mask = xoffset < C * num_groups
            col_mask = yoffset < HW
            rcmask = row_mask and col_mask
            offset = xoffset * HW + yoffset
            X_val = tl.load(X + offset, rcmask)
            x = tl.where(rcmask, X_val - mean, 0.0)
            var_base = var_base + x * x
    var = var_base / num_elements
    rstd = tl.libdevice.rsqrt(var + eps)

    X_val = tl.load(X_ptr, mask=xy_mask, other=0.0).to(tl.float32)
    x = tl.where(xy_mask, X_val - mean, 0.0)
    x_hat = x * rstd

    weight = tl.load(W_ptr, mask=wb_mask, other=0.0)[:, None]
    bias = tl.load(B_ptr, mask=wb_mask, other=0.0)[:, None]
    Y_val = x_hat * weight + bias

    tl.store(Y_ptr, Y_val, mask=xy_mask)
    tl.store(Mean_ptr, mean)
    tl.store(Rstd_ptr, rstd)


@libentry()
@triton.jit
def group_norm_backward_kernel(
    grad_y,
    X,
    W,
    Mean,
    Rstd,
    num_groups,
    group_size,
    grad_x,
    C,
    HW,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW

    Mean_ptr = Mean + pid
    Rstd_ptr = Rstd + pid

    rstd = tl.load(Rstd_ptr).to(tl.float32)
    mean = tl.load(Mean_ptr).to(tl.float32)

    grad_std = 0.0
    for xoffset in range(pid * group_size, pid * group_size + BLOCK_GROUP_SIZE, 1):
        row_mask = xoffset < C * num_groups
        w_offset = group * group_size + (xoffset - pid * group_size)
        weight = tl.load(W + w_offset, row_mask).to(tl.float32)
        for yoffset in range(0, HW, 1):
            col_mask = yoffset < HW
            rc_mask = row_mask and col_mask
            offset = xoffset * HW + yoffset
            dY_val = tl.load(grad_y + offset, rc_mask).to(tl.float32)
            X_val = tl.load(X + offset, rc_mask).to(tl.float32)
            dx_hat = weight * dY_val
            x = tl.where(rc_mask, X_val - mean, 0.0)
            grad_std = grad_std + dx_hat * x

    grad_centered_mean = 0.0
    for xoffset in range(pid * group_size, pid * group_size + BLOCK_GROUP_SIZE, 1):
        row_mask = xoffset < C * num_groups
        w_offset = group * group_size + (xoffset - pid * group_size)
        weight = tl.load(W + w_offset, row_mask).to(tl.float32)
        for yoffset in range(0, HW, 1):
            col_mask = yoffset < HW
            rc_mask = row_mask and col_mask
            offset = xoffset * HW + yoffset
            dY_val = tl.load(grad_y + offset, rc_mask).to(tl.float32)
            X_val = tl.load(X + offset, rc_mask).to(tl.float32)

            dx_hat = weight * dY_val
            x = tl.where(rc_mask, X_val - mean, 0.0)

            grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / (HW * group_size)
            grad_distance = 2 * x * grad_var
            grad_centered_mean += dx_hat * rstd + grad_distance
    grad_mean = -grad_centered_mean / num_elements

    for xoffset in range(pid * group_size, pid * group_size + BLOCK_GROUP_SIZE, 1):
        row_mask = xoffset < C * num_groups
        w_offset = group * group_size + (xoffset - pid * group_size)
        weight = tl.load(W + w_offset, row_mask).to(tl.float32)
        for yoffset in range(0, HW, 1):
            col_mask = yoffset < HW
            rc_mask = row_mask and col_mask
            offset = xoffset * HW + yoffset
            dY_val = tl.load(grad_y + offset, rc_mask).to(tl.float32)
            X_val = tl.load(X + offset, rc_mask).to(tl.float32)

            dx_hat = weight * dY_val
            x = tl.where(rc_mask, X_val - mean, 0.0)

            grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / (HW * group_size)
            grad_distance = 2 * x * grad_var
            grad_centered_mean = dx_hat * rstd + grad_distance
            grad_X = grad_centered_mean + grad_mean
            tl.store(grad_x + offset, grad_X, rc_mask)


@libentry()
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    num_groups,
    group_size,
    N,
    C,
    HW,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid // group_size

    dB_base = 0.0
    for xoffset in range(pid, pid + C + 1, C):
        x_mask = xoffset < C * BLOCK_N
        for yoffset in range(0, HW, 1):
            offset = xoffset * HW + yoffset
            grad_y = tl.load(dY + offset, x_mask).to(tl.float32)
            dB_base = dB_base + grad_y
    tl.store(dB + pid, dB_base)

    dW_base = 0.0
    for xoffset in range(pid, pid + C + 1, C):
        mr_offset = group + (xoffset // C) * num_groups
        rstd = tl.load(Rstd + mr_offset).to(tl.float32)
        mean = tl.load(Mean + mr_offset).to(tl.float32)
        x_mask = xoffset < C * BLOCK_N
        for yoffset in range(0, HW, 1):
            offset = xoffset * HW + yoffset
            grad_y = tl.load(dY + offset, x_mask).to(tl.float32)
            x = tl.load(X + offset, x_mask)
            x_f32 = x.to(tl.float32)
            dW_base = dW_base + (x_f32 - mean) * rstd * grad_y
    tl.store(dW + pid, dW_base)


class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, N, C, HW, num_groups, eps):
        logging.debug("GEMS GROUPNORM FORWARD")
        group_size = C // num_groups
        x = x.contiguous()
        if weight is not None:
            weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y = torch.empty_like(x)
        mean = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)
        rstd = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)
        grid = (N * num_groups,)

        with torch.cuda.device(x.device):
            group_norm_kernel[grid](
                x,
                y,
                weight,
                bias,
                mean,
                rstd,
                group_size,
                C,
                HW,
                num_groups,
                eps,
                BLOCK_GROUP_SIZE=triton.next_power_of_2(C // num_groups),
                BLOCK_HW_SIZE=triton.next_power_of_2(HW),
            )
        ctx.save_for_backward(x, weight, mean, rstd)
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
        (x, weight, mean, rstd) = ctx.saved_tensors
        num_groups = ctx.num_groups
        group_size = ctx.group_size
        N = ctx.N
        C = ctx.C
        HW = ctx.HW
        x_grad = torch.empty_like(x)
        weight_grad = torch.empty_like(weight)
        bias_grad = torch.empty_like(weight)
        grid = (N * num_groups,)
        with torch.cuda.device(x.device):
            group_norm_backward_kernel[grid](
                y_grad,
                x,
                weight,
                mean,
                rstd,
                num_groups,
                group_size,
                x_grad,
                C,
                HW,
                BLOCK_GROUP_SIZE=triton.next_power_of_2(C // num_groups),
                BLOCK_HW_SIZE=triton.next_power_of_2(HW),
            )
        weight_bias_backward_kernel[(C, 1, 1)](
            y_grad,
            x,
            mean,
            rstd,
            weight_grad,
            bias_grad,
            num_groups,
            group_size,
            N,
            C,
            HW,
            BLOCK_N=triton.next_power_of_2(N),
            BLOCK_HW=triton.next_power_of_2(HW),
        )
        return x_grad, weight_grad, bias_grad, None, None, None, None, None


def group_norm(x, weight, bias, N, C, HW, num_groups, eps):
    return GroupNorm.apply(x, weight, bias, N, C, HW, num_groups, eps)
