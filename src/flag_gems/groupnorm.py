import torch
import triton
import triton.language as tl
from .__libentry__ import libentry
import math

@libentry()
@triton.jit
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
    mini_batch = pid // num_groups
    group = pid % num_groups
    num_elements = group_size * HW
    batch_offset = mini_batch * num_groups * num_elements
    group_offset = batch_offset + group * num_elements
    Y_block_ptr = tl.make_block_ptr(
        Y + group_offset,
        shape=(group_size, HW),
        strides=(HW, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
        order=(1, 0),
    )
    X_block_ptr = tl.make_block_ptr(
        X + group_offset,
        shape=(group_size, HW),
        strides=(HW, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
        order=(1, 0),
    )
    rstd_block_ptr = tl.make_block_ptr(
        Rstd + mini_batch * num_groups,
        shape=(group_size,),
        strides=(1,),
        offsets=(group,),
        block_shape=(1,),
        order=(0,),
    )
    mean_block_ptr = tl.make_block_ptr(
        Mean + mini_batch * num_groups,
        shape=(group_size,),
        strides=(1,),
        offsets=(group,),
        block_shape=(1,),
        order=(0,),
    )
    # FIXME: Error when weight & bias are not None and dtype is float32
    weight_block_ptr = tl.make_block_ptr(
        W,
        shape=(C, ),
        strides=(1,),
        offsets=(group * group_size,),
        block_shape=(BLOCK_GROUP_SIZE,),
        order=(0,),
    )
    bias_block_ptr = tl.make_block_ptr(
        B,
        shape=(C,),
        strides=(1,),
        offsets=(group * group_size,),
        block_shape=(BLOCK_GROUP_SIZE,),
        order=(0,),
    )
    X_val = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero")
    mean = tl.sum(tl.view(X_val / num_elements, (1, BLOCK_GROUP_SIZE * BLOCK_HW_SIZE)), 1)
    group_mask = tl.arange(0, BLOCK_GROUP_SIZE) < group_size
    hw_mask = tl.arange(0, BLOCK_HW_SIZE) < HW
    mask = group_mask[:, None] & hw_mask[None, :]
    centered_mean = tl.where(mask, X_val - mean, 0)

    var = tl.sum(tl.view(centered_mean * centered_mean / num_elements, (1, BLOCK_GROUP_SIZE * BLOCK_HW_SIZE)), 1)
    rstd = tl.math.rsqrt(var + eps)
    Y_val = centered_mean * rstd

    weight = tl.load(weight_block_ptr, boundary_check=(0,))
    Y_val = Y_val * weight
    
    bias = tl.load(bias_block_ptr, boundary_check=(0, ))
    Y_val += bias

    tl.store(Y_block_ptr, Y_val.to(X_val.dtype), boundary_check=(0, 1))
    tl.store(rstd_block_ptr, rstd.to(X_val.dtype))
    tl.store(mean_block_ptr, mean.to(X_val.dtype))


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
    dW,
    dB,
    C,
    HW,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // num_groups
    group = pid % num_groups
    num_elements = group_size * HW
    batch_offset = batch * num_groups * num_elements
    group_offset = batch_offset + group * num_elements
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x + group_offset,
        shape=(group_size, HW),
        strides=(HW, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
        order=(1, 0),
    )
    grad_y_ptr = tl.make_block_ptr(
        grad_y + group_offset,
        shape=(group_size, HW),
        strides=(HW, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
        order=(1, 0),
    )
    x_block_ptr = tl.make_block_ptr(
        X + group_offset,
        shape=(group_size, HW),
        strides=(HW, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_GROUP_SIZE, BLOCK_HW_SIZE),
        order=(1, 0),
    )
    rstd_block_ptr = tl.make_block_ptr(
        Rstd + batch * num_groups,
        shape=(group_size,),
        strides=(1,),
        offsets=(group,),
        block_shape=(1,),
        order=(0,),
    )
    mean_block_ptr = tl.make_block_ptr(
        Mean + batch * num_groups,
        shape=(group_size,),
        strides=(1,),
        offsets=(group,),
        block_shape=(1,),
        order=(0,),
    )
    weight_block_ptr = tl.make_block_ptr(
        W,
        shape=(C,),
        strides=(1,),
        offsets=(group * group_size,),
        block_shape=(BLOCK_GROUP_SIZE,),
        order=(0,),
    )
    
    rstd = tl.load(rstd_block_ptr)
    mean = tl.load(mean_block_ptr)
    grad_Y = tl.load(grad_y_ptr, boundary_check=(0, 1))
    
    weight = tl.load(weight_block_ptr, boundary_check=(0,))
    grad_norm = weight * grad_Y

    X_val = tl.load(x_block_ptr, boundary_check=(0, 1))
    group_mask = tl.arange(0, BLOCK_GROUP_SIZE) < group_size
    hw_mask = tl.arange(0, BLOCK_HW_SIZE) < HW
    mask = group_mask[:, None] & hw_mask[None, :]
    centered_mean = tl.where(mask, X_val - mean, 0)
    grad_std = tl.sum(tl.view(grad_norm * centered_mean, (1, BLOCK_GROUP_SIZE * BLOCK_HW_SIZE)), 1)
    grad_var = grad_std * -(0.5 * rstd * rstd * rstd)/ (HW * group_size)
    grad_distance = 2 * centered_mean * grad_var
    grad_centered_mean = tl.where(mask, grad_norm * rstd + grad_distance, 0)
    grad_mean = -tl.sum(tl.view(grad_centered_mean, (1, BLOCK_GROUP_SIZE * BLOCK_HW_SIZE)), 1) / num_elements
    grad_X = grad_centered_mean + grad_mean
    tl.store(grad_x_block_ptr, grad_X.to(X_val.dtype), boundary_check=(0, 1))

    norm = centered_mean * rstd
    grad_weight = tl.sum(norm * grad_Y, 1)
    offset = batch * C + group * group_size
    dW_block_ptr = tl.make_block_ptr(
        dW + offset,
        shape=(group_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_GROUP_SIZE,),
        order=(0,),
    )
    
    tl.store(dW_block_ptr, grad_weight.to(X_val.dtype), boundary_check=(0,))

    grad_bias = tl.sum(grad_Y, 1)
    dB_block_ptr = tl.make_block_ptr(
        dB + offset,
        shape=(group_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_GROUP_SIZE,),
        order=(0,),
    )
    
    tl.store(dB_block_ptr, grad_bias.to(X_val.dtype), boundary_check=(0,))


class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, N, C, HW, num_groups, eps):
        if __debug__:
            print("GEMS GROUPNORM FORWARD")
        group_size = C // num_groups
        x = x.contiguous()
        if weight is not None:
            weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y = torch.empty_like(x)
        mean = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)
        rstd = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)
        grid = (N*C,)

        group_norm_kernel[grid](x, y, weight, bias, mean, rstd, group_size, C, HW, num_groups, eps, 
            BLOCK_GROUP_SIZE=triton.next_power_of_2(C // num_groups),
            BLOCK_HW_SIZE=triton.next_power_of_2(HW)
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
        if __debug__:
            print("GEMS GROUPNORM BACKWARD")
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
        grid = (N*C,)
        group_norm_backward_kernel[grid](
            y_grad, 
            x, 
            weight, 
            mean, 
            rstd, 
            num_groups, 
            group_size, 
            x_grad, 
            weight_grad, 
            bias_grad, 
            C, 
            HW, 
            BLOCK_GROUP_SIZE=triton.next_power_of_2(C // num_groups),
            BLOCK_HW_SIZE=triton.next_power_of_2(HW)
        )
        return x_grad, weight_grad, bias_grad, None, None, None, None, None


def group_norm(x, weight, bias, N, C, HW, num_groups, eps):
    return GroupNorm.apply(x, weight, bias, N, C, HW, num_groups, eps)
