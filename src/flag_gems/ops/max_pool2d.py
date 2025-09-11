import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def max_pool2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        return (numerator + stride - 1) // stride + 1
    else:
        return numerator // stride + 1


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64}, num_stages=2, num_warps=8),
    ],
    key=["out_h", "out_w", "kernel_h", "kernel_w", "stride_h", "stride_w"],
)
@triton.jit
def max_pool2d_forward_kernel_optimized(
    input_ptr,
    output_ptr,
    indices_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Pooling parameters
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # De-flatten program IDs to get N, C, and spatial block indices
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    # Calculate the coordinates of the output block this program is responsible for
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Initialize accumulators for max values and their indices
    dtype = input_ptr.type.element_ty
    max_val_acc = tl.full((BLOCK_H, BLOCK_W), get_dtype_min(dtype), dtype=dtype)
    max_idx_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int64)

    # Base pointer for the current batch and channel in the input tensor
    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    for kh in range(0, kernel_h):
        for kw in range(0, kernel_w):
            h_in = h_out_offsets[:, None] * stride_h - padding_h + kh * dilation_h
            w_in = w_out_offsets[None, :] * stride_w - padding_w + kw * dilation_w
            in_mask = (h_in >= 0) & (h_in < in_h) & (w_in >= 0) & (w_in < in_w)
            input_offset = h_in * in_stride_h + w_in * in_stride_w

            # Perform a vectorized load from global memory
            current_val = tl.load(
                input_base_ptr + input_offset, mask=in_mask, other=get_dtype_min(dtype)
            )

            # Calculate the flat index for each value in the block
            current_idx = h_in * in_w + w_in

            # Compare and update the max values and indices without branching
            is_new_max = current_val > max_val_acc
            max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
            max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_h * out_w
    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = (
        out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )
    indices_block_ptr = (
        indices_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    )

    # Create a mask to handle cases where output dimensions are not divisible by block size
    out_mask = (out_h_offsets[:, None] < out_h) & (out_w_offsets[None, :] < out_w)

    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)


@libentry()
@triton.jit
def max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Shape info
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
):
    # Each program instance handles one gradient from the output
    pid = tl.program_id(0)

    # De-flatten the program ID to get N, C indices
    n_idx = pid // (out_w * out_h * in_c)
    c_idx = (pid // (out_w * out_h)) % in_c

    grad_out_val = tl.load(grad_output_ptr + pid)
    max_idx_flat = tl.load(indices_ptr + pid)

    # Pointer to the start of the current batch and channel in the gradient input tensor
    grad_input_base_ptr = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    if max_idx_flat != -1:
        grad_input_offset = max_idx_flat
        # Atomically add the gradient to the correct location in grad_input
        tl.atomic_add(
            grad_input_base_ptr + grad_input_offset,
            grad_out_val,
        )


class MaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, ceil_mode):
        logger.debug("GEMS MAX_POOL2D FORWARD")
        input = input.contiguous()

        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size
        if stride is None or (isinstance(stride, (list, tuple)) and not stride):
            stride_h, stride_w = kernel_h, kernel_w
        elif isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
        if isinstance(padding, int):
            padding_h = padding_w = padding
        else:
            padding_h, padding_w = padding
        if isinstance(dilation, int):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        in_n, in_c, in_h, in_w = input.shape
        out_h = max_pool2d_output_size(
            in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
        )
        out_w = max_pool2d_output_size(
            in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
        )

        output = torch.empty(
            (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
        )
        indices = torch.empty(
            (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int64
        )

        if output.numel() > 0:
            # 2D Grid for the optimized kernel
            grid = lambda meta: (
                in_n * in_c,
                triton.cdiv(out_h, meta["BLOCK_H"])
                * triton.cdiv(out_w, meta["BLOCK_W"]),
            )

            max_pool2d_forward_kernel_optimized[grid](
                input,
                output,
                indices,
                input.stride(0),
                input.stride(1),
                input.stride(2),
                input.stride(3),
                in_c,
                in_h,
                in_w,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
            )

        ctx.save_for_backward(indices)
        ctx.in_shape = input.shape
        ctx.in_strides = (in_c * in_h * in_w, in_h * in_w, in_w, 1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS MAX_POOL2D BACKWARD")
        grad_output = grad_output.contiguous()
        (indices,) = ctx.saved_tensors
        in_shape = ctx.in_shape
        in_strides = ctx.in_strides

        in_n, in_c, in_h, in_w = in_shape
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        original_dtype = grad_output.dtype
        grad_input = torch.zeros(
            in_shape, device=grad_output.device, dtype=torch.float32
        )

        grid = (grad_output.numel(),)
        if grad_output.numel() > 0:
            max_pool2d_backward_kernel[grid](
                grad_output,
                indices,
                grad_input,
                in_strides[0],
                in_strides[1],
                in_strides[2],
                in_strides[3],
                in_c,
                in_h,
                in_w,
                out_h,
                out_w,
            )

        return grad_input.to(original_dtype), None, None, None, None, None


def max_pool2d(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return MaxPool2d.apply(self, kernel_size, stride, padding, dilation, ceil_mode)
