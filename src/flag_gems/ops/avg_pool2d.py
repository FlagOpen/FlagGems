import logging
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems import runtime

logger = logging.getLogger(__name__)


def avg_pool2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
) -> int:
    """
    Determines the output size of a avg pool2d operation.

    Args:
        in_size: Input size
        kernel_size: Kernel size
        stride: Stride
        padding: Implicit zero paddings on both sides of the input
        ceil_mode: When True, uses ceil instead of floor to compute output shape.\

    Returns:
        Output size of avg pool2d.
    """
    # To align with PyTorch's implementation
    # Different handling methods for imperfect coverage scenarios.
    effective_input_size = in_size + 2 * padding
    floor_size = (effective_input_size - kernel_size) // stride + 1

    if not ceil_mode:
        return floor_size

    if stride != kernel_size:
        output_size = (effective_input_size - kernel_size + stride - 1) // stride + 1
        return max(1, output_size)

    # For stride == kernel_size, check if last window covers only padding
    last_window_start = (floor_size - 1) * stride
    remaining_elements = (in_size + padding) - (last_window_start + stride)
    return floor_size + 1 if remaining_elements > 0 else floor_size


@triton.autotune(
    configs=runtime.get_tuned_config("avg_pool2d"),
    key=[
        "in_c",
        "input_height",
        "input_width",
        "kernel_height",
        "kernel_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "count_include_pad",
    ],
)
@triton.jit
def avg_pool2d_forward_kernel(
    input_ptr,
    output_ptr,
    in_c,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    count_include_pad: tl.constexpr,
    divisor_override: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    batch_pid = tl.program_id(axis=0)
    channel_pid = tl.program_id(axis=1)
    spatial_pid = tl.program_id(axis=2)

    channel_ptrs = channel_pid * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channel_ptrs < in_c

    spatial_ptrs = spatial_pid * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    output_h_ptrs = spatial_ptrs // output_width
    output_w_ptrs = spatial_ptrs % output_width

    # Maps output position to input window:
    input_h_start_ptrs = output_h_ptrs * stride_height - padding_height
    input_w_start_ptrs = output_w_ptrs * stride_width - padding_width

    input_h_end_ptrs = input_h_start_ptrs + kernel_height
    input_w_end_ptrs = input_w_start_ptrs + kernel_width

    if divisor_override is None:
        if count_include_pad:
            i_h_pad_e_ptrs = tl.minimum(input_h_end_ptrs, input_height + padding_height)
            i_w_pad_e_ptrs = tl.minimum(input_w_end_ptrs, input_width + padding_width)

            i_h_pad_s_ptrs = tl.maximum(input_h_start_ptrs, -padding_height)
            i_w_pad_s_ptrs = tl.maximum(input_w_start_ptrs, -padding_width)

            window_height = i_h_pad_e_ptrs - i_h_pad_s_ptrs
            window_width = i_w_pad_e_ptrs - i_w_pad_s_ptrs
        else:
            i_h_e_ptrs_clamped = tl.minimum(input_h_end_ptrs, input_height)
            i_w_e_ptrs_clamped = tl.minimum(input_w_end_ptrs, input_width)

            i_h_s_ptrs_clamped = tl.maximum(input_h_start_ptrs, 0)
            i_w_s_ptrs_clamped = tl.maximum(input_w_start_ptrs, 0)

            window_height = i_h_e_ptrs_clamped - i_h_s_ptrs_clamped
            window_width = i_w_e_ptrs_clamped - i_w_s_ptrs_clamped

        effective_divisor = tl.maximum(window_height * window_width, 1)
    else:
        effective_divisor = tl.maximum(divisor_override, 1)

    # Accumulator: shape [BLOCK_C, BLOCK_SPATIAL]
    acc = tl.zeros((BLOCK_C, BLOCK_SPATIAL), dtype=tl.float32)

    for kh in range(0, kernel_height):
        for kw in range(0, kernel_width):
            h = input_h_start_ptrs + kh
            w = input_w_start_ptrs + kw

            spatial_mask = (h >= 0) & (h < input_height) & (w >= 0) & (w < input_width)

            input_offset = (
                batch_pid * in_c * input_height * input_width
                + channel_ptrs[:, None] * input_height * input_width
                + h[None, :] * input_width
                + w[None, :]
            )

            acc += tl.load(
                input_ptr + input_offset,
                mask=channel_mask[:, None] & spatial_mask[None, :],
                other=0.0,
            )

    output_val = acc / effective_divisor[None, :]
    output_offset = (
        batch_pid * in_c * output_height * output_width
        + channel_ptrs[:, None] * output_height * output_width
        + output_h_ptrs[None, :] * output_width
        + output_w_ptrs[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        output_val,
        mask=channel_mask[:, None]
        & (spatial_ptrs < output_height * output_width)[None, :],
    )


def avg_pool2d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> torch.Tensor:
    """
    2D pooling operation interface.

    Args:
        input: Input tensor of shape (minibatch, in_channels, iH, iW)
        kernel_size: Size of the pooling region. Can be a single number or a tuple (kH, kW)
        stride: Stride of the pooling operation. Defaults to kernel_size if None
        padding: Implicit zero paddings on both sides of the input
        ceil_mode: When True, uses ceil instead of floor to compute output shape
        count_include_pad: When True, includes zero-padding in averaging calculation
        divisor_override: If specified, used as divisor instead of pooling region size

    Returns:
        Output tensor after pooling operation
    """
    logger.debug("GEMS AVG_POOL2D")
    assert input.ndim == 4, f"Input must be 4D, received shape {input.shape}"
    if isinstance(kernel_size, (list, tuple)):
        kernel_height, kernel_width = kernel_size
    else:
        kernel_height = kernel_width = kernel_size

    if stride is None:
        stride_height, stride_width = kernel_height, kernel_width
    elif isinstance(stride, (list, tuple)):
        stride_height, stride_width = stride
    else:
        stride_height = stride_width = stride

    if isinstance(padding, (list, tuple)):
        padding_height, padding_width = padding
    else:
        padding_height = padding_width = padding

    in_n, in_c, input_height, input_width = input.shape

    output_height = avg_pool2d_output_size(
        input_height,
        kernel_height,
        stride_height,
        padding_height,
        ceil_mode,
    )
    output_width = avg_pool2d_output_size(
        input_width,
        kernel_width,
        stride_width,
        padding_width,
        ceil_mode,
    )

    output = torch.empty(
        (in_n, in_c, output_height, output_width),
        dtype=input.dtype,
        device=input.device,
    )

    grid = lambda META: (
        in_n,
        triton.cdiv(in_c, META["BLOCK_C"]),
        triton.cdiv(output_height * output_width, META["BLOCK_SPATIAL"]),
    )

    avg_pool2d_forward_kernel[grid](
        input,
        output,
        in_c,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        count_include_pad,
        divisor_override,
    )
    return output
