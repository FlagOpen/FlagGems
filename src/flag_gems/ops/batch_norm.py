import logging
from typing import Dict

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton import next_power_of_2

from .. import runtime
from ..utils import libentry
from ..utils.type_utils import get_accumulator_dtype


def make_3d_for_bn(input: Tensor) -> Tensor:
    """
    Converts the input to a 3D view for batch normalization.

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)

    elif input.ndim >= 4:
        input = input.flatten(2, -1)

    return input


def BLOCK_SIZE_SPATIAL_heuristic(args: Dict) -> int:
    """
    Approximates an appropriate spatial block size for batch normalization
    using a heuristic.

    Args:
        args: Arguments to batch normalization kernel.

    Returns:
        Appropriate spatial block size.
    """
    # Preferrably, the batch and spatial dimensions would both be loaded,
    # and normalization would be conducted in one step.
    # However, for large inputs, that is not feasible given memory constraints.
    # Thus, a maximum of 16384 elements are loaded at once.
    BLOCK_SIZE_BATCH = next_power_of_2(args["batch_dim"])
    BLOCK_SIZE_SPATIAL = next_power_of_2(args["spatial_dim"])
    return min(BLOCK_SIZE_SPATIAL, max(1, 2**14 // BLOCK_SIZE_BATCH))


@libentry()
@triton.autotune(
    configs=runtime.get_triton_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
    restore_value=["running_mean_pointer", "running_var_pointer"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": lambda args: next_power_of_2(args["batch_dim"]),
        "BLOCK_SIZE_SPATIAL": BLOCK_SIZE_SPATIAL_heuristic,
    }
)
@triton.jit
def batch_norm_forward_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    mean_pointer,
    inv_std_pointer,
    output_pointer,
    running_mean_pointer,
    running_var_pointer,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    momentum,
    eps,
    affine: tl.constexpr,
    save_stats: tl.constexpr,
    is_train: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim

    # traning mode default track_running_stat
    if is_train:
        count = 0
        mean = 0.0
        var = 0.0

        for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
            spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(
                0, BLOCK_SIZE_SPATIAL
            )
            spatial_mask = spatial_offset < spatial_dim

            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)

            spatial_count = min(
                BLOCK_SIZE_SPATIAL, spatial_dim - block_ind * BLOCK_SIZE_SPATIAL
            )
            curr_count = spatial_count * batch_dim
            count += curr_count

            prev_mean = mean
            mean += (tl.sum(curr_input) - curr_count * mean) / count
            deltas = tl.where(
                batch_mask[:, None] & spatial_mask[None, :],
                (curr_input - mean) * (curr_input - prev_mean),
                0.0,
            )
            var += tl.sum(deltas)

        var /= count
        inv_std = tl.rsqrt(var + eps)

        if save_stats:
            tl.store(feat_pid + mean_pointer, mean)
            tl.store(feat_pid + inv_std_pointer, inv_std)

        running_mean_pointer += feat_pid
        running_var_pointer += feat_pid

        running_mean = tl.load(running_mean_pointer)
        running_var = tl.load(running_var_pointer)

        n = batch_dim * spatial_dim
        tl.store(running_mean_pointer, (1 - momentum) * running_mean + momentum * mean)
        tl.store(
            running_var_pointer,
            (1 - momentum) * running_var + momentum * var * n / (n - 1),
        )

    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        inv_std = tl.rsqrt(tl.load(feat_pid + running_var_pointer) + eps)

    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        bias = tl.load(feat_pid + bias_pointer)

    else:
        weight = 1.0
        bias = 0.0

    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(
            0, BLOCK_SIZE_SPATIAL
        )
        spatial_mask = spatial_offset < spatial_dim

        curr_input_pointer = (
            input_pointer
            + input_feat_stride * feat_pid
            + input_batch_stride * batch_offset[:, None]
            + input_spatial_stride * spatial_offset[None, :]
        )
        curr_output_pointer = (
            output_pointer
            + output_feat_stride * feat_pid
            + output_batch_stride * batch_offset[:, None]
            + output_spatial_stride * spatial_offset[None, :]
        )

        curr_input = tl.load(
            curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)
        output = weight * (curr_input - mean) * inv_std + bias

        tl.store(
            curr_output_pointer,
            output,
            mask=batch_mask[:, None] & spatial_mask[None, :],
        )


@triton.autotune(
    configs=runtime.get_triton_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_BATCH": lambda args: next_power_of_2(args["batch_dim"]),
        "BLOCK_SIZE_SPATIAL": BLOCK_SIZE_SPATIAL_heuristic,
    }
)
@triton.jit
def batch_norm_backward_kernel(
    output_grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    weight_pointer,
    input_grad_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_feat_stride,
    output_grad_spatial_stride,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    input_grad_spatial_stride,
    affine: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim

    mean = tl.load(feat_pid + mean_pointer).to(tl.float32)
    inv_std = tl.load(feat_pid + inv_std_pointer).to(tl.float32)

    term1 = tl.zeros([BLOCK_SIZE_BATCH, BLOCK_SIZE_SPATIAL], dtype=tl.float32)
    term2 = tl.zeros([BLOCK_SIZE_BATCH, BLOCK_SIZE_SPATIAL], dtype=tl.float32)

    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(
            0, BLOCK_SIZE_SPATIAL
        )
        spatial_mask = spatial_offset < spatial_dim

        curr_output_grad_pointer = (
            output_grad_pointer
            + output_grad_feat_stride * feat_pid
            + output_grad_batch_stride * batch_offset[:, None]
            + output_grad_spatial_stride * spatial_offset[None, :]
        )
        curr_input_pointer = (
            input_pointer
            + input_feat_stride * feat_pid
            + input_batch_stride * batch_offset[:, None]
            + input_spatial_stride * spatial_offset[None, :]
        )

        curr_input = tl.load(
            curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)

        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(
            curr_output_grad_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)

        term1 += curr_pre_lin * curr_output_grad
        term2 += curr_output_grad

    term1 = tl.sum(term1)
    term2 = tl.sum(term2)

    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        weight_grad = 0.0
        bias_grad = 0.0

    else:
        weight = 1.0

    count = batch_dim * spatial_dim

    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(
            0, BLOCK_SIZE_SPATIAL
        )
        spatial_mask = spatial_offset < spatial_dim

        curr_output_grad_pointer = (
            output_grad_pointer
            + output_grad_feat_stride * feat_pid
            + output_grad_batch_stride * batch_offset[:, None]
            + output_grad_spatial_stride * spatial_offset[None, :]
        )
        curr_input_pointer = (
            input_pointer
            + input_feat_stride * feat_pid
            + input_batch_stride * batch_offset[:, None]
            + input_spatial_stride * spatial_offset[None, :]
        )
        curr_input_grad_pointer = (
            input_grad_pointer
            + input_grad_feat_stride * feat_pid
            + input_grad_batch_stride * batch_offset[:, None]
            + input_grad_spatial_stride * spatial_offset[None, :]
        )

        curr_input = tl.load(
            curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(
            curr_output_grad_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
        ).to(tl.float32)
        curr_input_grad = (
            inv_std
            * weight
            * (curr_output_grad - (term1 * curr_pre_lin + term2) / count)
        )
        tl.store(
            curr_input_grad_pointer,
            curr_input_grad,
            mask=batch_mask[:, None] & spatial_mask[None, :],
        )

        if affine:
            weight_grad += tl.sum(curr_pre_lin * curr_output_grad)
            bias_grad += tl.sum(curr_output_grad)

    if affine:
        tl.store(feat_pid + weight_grad_pointer, weight_grad)
        tl.store(feat_pid + bias_grad_pointer, bias_grad)


class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight=None,
        bias=None,
        running_mean=None,  # self.running_mean if not self.training or self.track_running_state else None
        running_var=None,
        training=False,  # (self.running_mean is None) and (self.running_var is None)
        momentum=0.1,
        eps=1e-05,
        cudnn_enable=True,
    ):
        logging.debug("GEMS BATCHNORM FORWARD")

        input_3d = make_3d_for_bn(input)

        affine = weight is not None and bias is not None
        requires_grad = (
            input.requires_grad
            or (affine and weight.requires_grad)
            or (affine and bias.requires_grad)
        )

        batch_dim, feat_dim, spatial_dim = input_3d.shape
        output = torch.empty_like(input_3d)

        if requires_grad:
            acc_type = get_accumulator_dtype(input.dtype)
            mean = torch.empty(feat_dim, device=input.device, dtype=acc_type)
            inv_std = torch.empty(feat_dim, device=input.device, dtype=acc_type)

        else:
            mean = inv_std = None

        running_mean = input if running_mean is None else running_mean
        running_var = input if running_var is None else running_var

        # Launches 1D grid where each program operates over one feature.
        grid = lambda _: (feat_dim,)
        batch_norm_forward_kernel[grid](
            input_3d,
            weight,
            bias,
            mean,
            inv_std,
            output,
            running_mean,
            running_var,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            momentum,
            eps,
            affine=affine,
            save_stats=requires_grad,
            is_train=training,
        )

        ctx.affine = affine
        if requires_grad:
            ctx.save_for_backward(input, mean, inv_std, weight)

        return output.view_as(input)

    @staticmethod
    def backward(ctx, output_grad):
        logging.debug("GEMS BATCHNORM BACKWARD")
        (input, mean, inv_std, weight) = ctx.saved_tensors
        input_3d = make_3d_for_bn(input)
        output_grad_3d = make_3d_for_bn(output_grad)

        batch_dim, feat_dim, spatial_dim = input_3d.shape
        input_grad = torch.empty_like(input_3d)

        if ctx.affine:
            weight_grad = torch.empty((feat_dim,), device=input.device)
            bias_grad = torch.empty_like(weight_grad)

        else:
            weight_grad = bias_grad = None

        # Launches 1D grid where each program operates over one feature.
        grid = lambda _: (feat_dim,)
        batch_norm_backward_kernel[grid](
            output_grad_3d,
            input_3d,
            mean,
            inv_std,
            weight,
            input_grad,
            weight_grad,
            bias_grad,
            batch_dim,
            spatial_dim,
            *output_grad_3d.stride(),
            *input_3d.stride(),
            *input_grad.stride(),
            affine=ctx.affine,
        )

        # Pads output with None because a gradient is necessary for
        # all input arguments.
        return (
            input_grad.view_as(input),
            weight_grad,
            bias_grad,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def batch_norm(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
    cudnn_enable=True,
):
    return BatchNorm.apply(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cudnn_enable,
    )
