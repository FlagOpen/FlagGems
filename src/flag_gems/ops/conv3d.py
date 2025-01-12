import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_SIZE": 128}, num_stages=4, num_warps=8),
    ],
    key=["N", "C", "D", "H", "W", "K"],
)
@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    C,
    D,
    H,
    W,
    K,
    T,
    R,
    S,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dilation_d,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_idx = tl.program_id(1)
    k_idx = tl.program_id(2)

    # Compute output dimensions
    out_d = (D + 2 * pad_d - dilation_d * (T - 1) - 1) // stride_d + 1
    out_h = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1

    # Compute start and end indices for the block
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, out_d * out_h * out_w)

    # Number of channels per group
    C_per_group = C // groups
    K_per_group = K // groups

    # Loop over the block
    for idx in range(block_start, block_end):
        # Compute the output indices
        od = idx // (out_h * out_w)
        oh = (idx % (out_h * out_w)) // out_w
        ow = idx % out_w

        # Compute the corresponding input indices
        id = od * stride_d - pad_d
        ih = oh * stride_h - pad_h
        iw = ow * stride_w - pad_w

        # Initialize the output value
        value = tl.zeros((), dtype=tl.float32)

        # Perform convolution
        group_id = k_idx // K_per_group
        for t in range(T):
            for r in range(R):
                for s in range(S):
                    for c in range(C_per_group):
                        input_d = id + t * dilation_d
                        input_h = ih + r * dilation_h
                        input_w = iw + s * dilation_w

                        # Check if the input indices are valid
                        in_bounds = (
                            (0 <= input_d)
                            & (input_d < D)
                            & (0 <= input_h)
                            & (input_h < H)
                            & (0 <= input_w)
                            & (input_w < W)
                        )

                        if in_bounds:
                            input_idx = (
                                n_idx * C * D * H * W
                                + (group_id * C_per_group + c) * D * H * W
                                + input_d * H * W
                                + input_h * W
                                + input_w
                            )
                            weight_idx = (
                                k_idx * C_per_group * T * R * S
                                + c * T * R * S
                                + t * R * S
                                + r * S
                                + s
                            )
                            input_val = tl.load(
                                input_ptr + input_idx, mask=in_bounds, other=0.0
                            ).to(tl.float32)
                            weight_val = tl.load(weight_ptr + weight_idx).to(tl.float32)
                            value += input_val * weight_val

        # Store the result in the output tensor
        output_idx = (
            n_idx * K * out_d * out_h * out_w
            + k_idx * out_d * out_h * out_w
            + od * out_h * out_w
            + oh * out_w
            + ow
        )
        tl.store(output_ptr + output_idx, value)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Implements a 3D convolution with groups support using Triton.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, D, H, W).
        weight (torch.Tensor): Weight tensor of shape (K, C/groups, T, R, S).
        bias (torch.Tensor, optional): Bias tensor of shape (K,).
        stride (int or tuple): Stride of the convolution. Default: 1.
        padding (int or tuple): Padding added to all three dimensions. Default: 0.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input channels to output channels. Default: 1.

    Returns:
        torch.Tensor: Output tensor.
    """
    # Input shape: (N, C, D, H, W)
    # Weight shape: (K, C/groups, T, R, S)
    N, C, D, H, W = input.shape
    K, C_per_group, T, R, S = weight.shape

    # Expand parameters to 3D
    stride_d, stride_h, stride_w = (
        (stride, stride, stride) if isinstance(stride, int) else stride
    )
    pad_d, pad_h, pad_w = (
        (padding, padding, padding) if isinstance(padding, int) else padding
    )
    dilation_d, dilation_h, dilation_w = (
        (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
    )

    # Compute output dimensions
    out_d = (D + 2 * pad_d - dilation_d * (T - 1) - 1) // stride_d + 1
    out_h = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
    out_w = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1

    # Allocate output tensor
    output = torch.zeros(
        (N, K, out_d, out_h, out_w), device=input.device, dtype=input.dtype
    )

    # Triton grid configuration
    grid = lambda META: (
        triton.cdiv(
            out_d * out_h * out_w, META["BLOCK_SIZE"]
        ),  # Number of output elements divided into blocks
        N,  # Number of batches
        K,  # Number of output channels
    )

    # Launch kernel
    conv3d_kernel[grid](
        input,
        weight,
        output,
        N,
        C,
        D,
        H,
        W,
        K,
        T,
        R,
        S,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        groups,
    )

    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)

    return output
