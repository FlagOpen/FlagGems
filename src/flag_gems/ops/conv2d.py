import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


def conv2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> int:
    """
    Determines the output size of a 2D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.

    Returns:
        Output size of 2D convolution.
    """
    return (in_size + 2 * padding - kernel_size) // stride + 1


def conv2d_forward_config(
    BLOCK_NI_HO_WO: int,
    BLOCK_CI: int,
    BLOCK_CO: int,
    n_warps: int,
    num_stages: int,
) -> triton.Config:
    """
    Creates a triton.Config object for conv2d_forward_kernel
    given meta-parameters for auto-tuning.

    Args:
        BLOCK_NI_HO_WO: Block size across the input n, output height, and
                        output width dimensions.
        BLOCK_CI: Block size across the input c dimension.
        BLOCK_CO: Block size across the output c dimension.
        n_warps: Number of warps to use for the kernel when compiled for GPUs.
        num_stages: Number of stages the compiler uses to software-pipeline.

    Returns:
        Kernel configuration.
    """
    return triton.Config(
        {"BLOCK_NI_HO_WO": BLOCK_NI_HO_WO, "BLOCK_CI": BLOCK_CI, "BLOCK_CO": BLOCK_CO},
        num_warps=n_warps,
        num_stages=num_stages,
    )


@libentry()
@triton.autotune(
    configs=[
        conv2d_forward_config(128, 32, 128, n_warps=8, num_stages=2),
        conv2d_forward_config(256, 32, 128, n_warps=8, num_stages=2),
        conv2d_forward_config(256, 32, 64, n_warps=4, num_stages=4),
        conv2d_forward_config(256, 64, 32, n_warps=4, num_stages=4),
        conv2d_forward_config(256, 64, 16, n_warps=2, num_stages=4),
        conv2d_forward_config(256, 64, 128, n_warps=8, num_stages=4),
        conv2d_forward_config(128, 32, 64, n_warps=4, num_stages=4),
        conv2d_forward_config(128, 32, 16, n_warps=4, num_stages=4),
        conv2d_forward_config(128, 128, 128, n_warps=8, num_stages=3),
        conv2d_forward_config(64, 128, 128, n_warps=4, num_stages=4),
        conv2d_forward_config(64, 128, 64, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 64, 64, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 32, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 16, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(64, 16, 16, n_warps=2, num_stages=4),
        conv2d_forward_config(32, 32, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(32, 16, 32, n_warps=2, num_stages=4),
        conv2d_forward_config(32, 16, 16, n_warps=2, num_stages=4),
    ],
    key=[
        "in_n",
        "kernel_c",
        "in_height",
        "in_width",
        "out_c",
        "out_height",
        "out_width",
        "kernel_height",
        "kernel_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "groups",
    ],
)
@triton.jit
def conv2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    in_n,
    in_height,
    in_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    kernel_c: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_ni_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # caculate in_n out_height out_weight value in kernel
    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_width
    in_n_point_value = ni_ho_offset // out_height
    output_height_point_value = ni_ho_offset % out_height
    output_width_point_value = ni_ho_wo_offset % out_width

    # Load the input and weight pointers. input and weight are of shape
    # [in_n, groups, in_c, in_height, in_width] and [groups, out_c, in_c, kernel_height, kernel_width]
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * kernel_c
    )[:, None]
    weight_pointer += (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )[None, :]

    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)

    for h in range(kernel_height):
        for w in range(kernel_width):
            for c in range(0, kernel_c, BLOCK_CI):
                input_c_offset = c + tl.arange(0, BLOCK_CI)
                input_height_offset = (
                    h - padding_height + stride_height * output_height_point_value
                )
                input_width_offset = (
                    w - padding_width + stride_width * output_width_point_value
                )

                curr_input_pointer = (
                    input_pointer
                    + (input_c_stride * input_c_offset)[None, :]
                    + (input_height_stride * input_height_offset)[:, None]
                    + (input_width_stride * input_width_offset)[:, None]
                )
                curr_weight_pointer = (
                    weight_pointer
                    + (weight_c_stride * input_c_offset)[:, None]
                    + (weight_height_stride * h)[None, :]
                    + (weight_width_stride * w)[None, :]
                )

                input_mask = (
                    (in_n_point_value < in_n)[:, None]
                    & (input_c_offset < kernel_c)[None, :]
                    & (0 <= input_height_offset)[:, None]
                    & (input_height_offset < in_height)[:, None]
                    & (0 <= input_width_offset)[:, None]
                    & (input_width_offset < in_width)[:, None]
                )
                weight_mask = (input_c_offset < kernel_c)[:, None] & (
                    output_c_offset < out_per_group_c
                )[None, :]

                input_block = tl.load(curr_input_pointer, mask=input_mask)
                weight_block = tl.load(curr_weight_pointer, mask=weight_mask)

                accum += tl.dot(input_block, weight_block, allow_tf32=False)

    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    tl.store(output_pointer, accum, mask=output_mask)


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        logging.debug("GEMS CONV2D")
        assert weight.ndim == 4, "Weights must be 4D, received shape {weight.shape}"
        assert (
            bias is None or bias.ndim == 1
        ), "Bias must be 1D, received shape {bias.shape}"

        assert (
            input.shape[1] == groups * weight.shape[1]
        ), "Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups"
        assert (
            bias is None or weight.shape[0] == bias.shape[0]
        ), "Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

        if isinstance(stride, (list, tuple)):
            stride_height, stride_width = stride
        else:
            stride_height = stride_width = stride

        if isinstance(padding, (list, tuple)):
            padding_height, padding_width = padding
        else:
            padding_height = padding_width = padding

        in_n, _, in_height, in_width = input.shape
        out_c, kernel_c, kernel_height, kernel_width = weight.shape
        out_height = conv2d_output_size(
            in_height, kernel_height, stride_height, padding_height
        )
        out_width = conv2d_output_size(
            in_width, kernel_width, stride_width, padding_width
        )

        output_dtype = input.dtype
        output = torch.empty(
            (in_n, out_c, out_height, out_width),
            device=input.device,
            dtype=output_dtype,
        )

        # BLOCK_NI_HO_WO along the in_n, out_height, and out_width dimensions,
        # BLOCK_CO along the out_c,
        # one group per cat
        grid = lambda META: (
            triton.cdiv(in_n * out_height * out_width, META["BLOCK_NI_HO_WO"]),
            triton.cdiv(out_c, META["BLOCK_CO"]),
            groups,
        )
        conv2d_forward_kernel[grid](
            input,
            weight,
            output,
            in_n,
            in_height,
            in_width,
            out_c,
            out_height,
            out_width,
            *input.stride(),
            *weight.stride(),
            *output.stride(),
            kernel_c,
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            groups=groups,
        )

        if bias is not None:
            # Adding bias in the kernel becomes buggy when groups != 1.
            output += bias.view(1, -1, 1, 1)

        requires_grad = (
            input.requires_grad
            or weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )

        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.groups = groups
        ctx.bias_requires_grad = False if bias is None else bias.requires_grad
        ctx.output_dtype = output_dtype
        if requires_grad:
            ctx.save_for_backward(input, weight)

        return output


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return Conv2d.apply(input, weight, bias, stride, padding, dilation, groups)
