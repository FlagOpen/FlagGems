import logging
from typing import Optional, Tuple, List

import torch
import triton
import triton.language as tl

from .. import runtime
from ..runtime import device, torch_device_fn
from ..utils import triton_lang_extension as tle

device = device.name
logger = logging.getLogger(__name__)

# @triton.autotune(
#     configs=runtime.get_tuned_config("upsample_bilinear2d"),
#     key=["H_out", "W_out"],
#     use_cuda_graph=True,
# )
# @triton.jit
# def upsample_bilinear2d_kernel(
#     input_ptr, output_ptr, 
#     H_in, W_in,
#     H_out, W_out,
#     align_corners,
#     input_stride_b: tl.constexpr, input_stride_c: tl.constexpr, input_stride_h: tl.constexpr, input_stride_w: tl.constexpr,
#     output_stride_b: tl.constexpr, output_stride_c: tl.constexpr, output_stride_h: tl.constexpr, output_stride_w: tl.constexpr,
#     BLOCK_SIZE_H: tl.constexpr = 128,
#     BLOCK_SIZE_W: tl.constexpr = 128,
# ):
#     pid_w = tl.program_id(axis=0)
#     pid_h = tl.program_id(axis=1)
#     pid_bc = tl.program_id(axis=2)
#     # b/c offset 
#     batch = (pid_bc / output_stride_b).to(tl.uint32)
#     channel = (pid_bc % output_stride_b).to(tl.uint32)
#     # offset
#     offset_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)[:, None]
#     offset_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)[None, :]
#     mask = (offset_h < H_out) & (offset_w < W_out)
#     # Coordinate Mapping
#     if align_corners == True:
#         x = offset_w.to(tl.float32) * (W_in - 1) / (W_out - 1)
#         y = offset_h.to(tl.float32) * (H_in - 1) / (H_out - 1)
#     else:
#         x = ((offset_w.to(tl.float32) + 0.5) * W_in / W_out) - 0.5
#         y = ((offset_h.to(tl.float32) + 0.5) * H_in / H_out) - 0.5
#     # Determine the coordinates of neighboring points
#     x1 = tl.maximum(tl.floor(x).to(tl.uint32), 0)
#     x2 = tl.minimum(tl.ceil(x).to(tl.uint32), W_in - 1)
#     y1 = tl.maximum(tl.floor(y).to(tl.uint32), 0)
#     y2 = tl.minimum(tl.ceil(y).to(tl.uint32), H_in - 1)
#     # Obtain neighboring pixel values
#     input_bc_ptrs = input_ptr + batch*input_stride_b + channel*input_stride_c
#     Q11 = tl.load(input_bc_ptrs + y1*input_stride_h + x1, mask=mask, other=0.0).to(tl.float32)
#     Q21 = tl.load(input_bc_ptrs + y1*input_stride_h + x2, mask=mask, other=0.0).to(tl.float32)
#     Q12 = tl.load(input_bc_ptrs + y2*input_stride_h + x1, mask=mask, other=0.0).to(tl.float32)
#     Q22 = tl.load(input_bc_ptrs + y2*input_stride_h + x2, mask=mask, other=0.0).to(tl.float32)
#     # Calculate weights
#     dx = x - x1
#     dy = y - y1
#     # bilinear interpolation
#     P = Q11 * (1 - dx) * (1 - dy) + Q21 * dx * (1 - dy) + Q12 * (1 - dx) * dy + Q22 * dx * dy 
#     # # save result
#     output_bc_ptrs = output_ptr + batch*output_stride_b + channel*output_stride_c + offset_h*output_stride_h + offset_w
#     tl.store(output_bc_ptrs, P, mask=mask)


# def upsample_bilinear2d(
#         input: torch.Tensor, 
#         output_size: Optional[List[int]] = None,
#         align_corners : bool = True, 
#         scale_factors : Optional[List[float]] = None
# ) -> torch.Tensor:
#     assert isinstance(input, torch.Tensor), "Input must be a PyTorch Tensor."
#     assert input.dim() == 4, f"Input must be 4D tensor (got {input.dim()}D) - (N, C, H, W)."
#     if output_size is not None:
#         assert len(output_size) == 2, f"size must have 2 elements (H, W), got {len(output_size)}."
#         assert all(isinstance(d, int) and d > 0 for d in output_size), f"Each element in size must be a positive integer, got {output_size}"
#     if scale_factors is not None:
#         assert all(isinstance(d, float) and d > 0 for d in scale_factors), f"Each element in size must be a positive float, got {scale_factors}"
#     if output_size is not None and scale_factors is not None:
#         raise ValueError("Only one of 'size' or 'scale_factor' can be provided.")
#     if output_size is None and scale_factors is None:
#         raise ValueError("Either 'size' or 'scale_factor' must be provided")
#     N, C, H_in, W_in = input.shape
#     if output_size is not None:
#         H_out, W_out = output_size
#     else:
#         H_out = int(H_in * scale_factors[0])
#         W_out = int(W_in * scale_factors[1])
#     H_out, W_out = output_size
   
#     output = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=input.device)
#     # grid = lambda META: (N * C, triton.cdiv(H_out, META['BLOCK_SIZE_H']), triton.cdiv(W_out, META['BLOCK_SIZE_W']))    
#     grid = lambda META: (triton.cdiv(W_out, META['BLOCK_SIZE_W']), triton.cdiv(H_out, META['BLOCK_SIZE_H']), N * C)   
#     upsample_bilinear2d_kernel[grid](
#         input, output, 
#         H_in, W_in, 
#         H_out, W_out, 
#         align_corners,
#         *input.stride(), 
#         *output.stride(),
#     )
#     return output









@triton.autotune(
    configs=runtime.get_tuned_config("upsample_bilinear2d"),
    key=["H_out", "W_out"],
    use_cuda_graph=True,
)
@triton.jit
def upsample_bilinear2d_kernel(
    input_ptr, output_ptr, 
    H_in, W_in,
    H_out, W_out,
    batch_size, channel_num,
    align_corners,
    input_stride_b: tl.constexpr, input_stride_c: tl.constexpr, input_stride_h: tl.constexpr, input_stride_w: tl.constexpr,
    output_stride_b: tl.constexpr, output_stride_c: tl.constexpr, output_stride_h: tl.constexpr, output_stride_w: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr = 128,
    BLOCK_SIZE_W: tl.constexpr = 128,
):
    pid = tl.program_id(axis=0)

    total_blocks_w = tl.cdiv(W_out, BLOCK_SIZE_W)
    total_blocks_h = tl.cdiv(H_out, BLOCK_SIZE_H)
    total_blocks_per_bc = total_blocks_h * total_blocks_w
    total_bc = tl.cdiv(pid + 1, total_blocks_per_bc)

    bc = pid // total_blocks_per_bc
    block_idx = pid % total_blocks_per_bc

    block_h_idx = block_idx // total_blocks_w
    block_w_idx = block_idx % total_blocks_w
    # batch / channel offset
    batch = bc // channel_num
    channel = bc % channel_num
    # offset
    offset_h = block_h_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)[:, None]
    offset_w = block_w_idx * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)[None, :]
    mask = (offset_h < H_out) & (offset_w < W_out)
    # Coordinate Mapping
    if align_corners == True:
        x = offset_w.to(tl.float32) * (W_in - 1) / (W_out - 1)
        y = offset_h.to(tl.float32) * (H_in - 1) / (H_out - 1)
    else:
        x = ((offset_w.to(tl.float32) + 0.5) * W_in / W_out) - 0.5
        y = ((offset_h.to(tl.float32) + 0.5) * H_in / H_out) - 0.5
    # Determine the coordinates of neighboring points
    x1 = tl.maximum(tl.floor(x).to(tl.uint32), 0)
    x2 = tl.minimum(tl.ceil(x).to(tl.uint32), W_in - 1)
    y1 = tl.maximum(tl.floor(y).to(tl.uint32), 0)
    y2 = tl.minimum(tl.ceil(y).to(tl.uint32), H_in - 1)
    # Obtain neighboring pixel values
    input_bc_ptrs = input_ptr + batch*input_stride_b + channel*input_stride_c
    Q11 = tl.load(input_bc_ptrs + y1*input_stride_h + x1, mask=mask, other=0.0).to(tl.float32)
    Q21 = tl.load(input_bc_ptrs + y1*input_stride_h + x2, mask=mask, other=0.0).to(tl.float32)
    Q12 = tl.load(input_bc_ptrs + y2*input_stride_h + x1, mask=mask, other=0.0).to(tl.float32)
    Q22 = tl.load(input_bc_ptrs + y2*input_stride_h + x2, mask=mask, other=0.0).to(tl.float32)
    # Calculate weights
    dx = x - x1
    dy = y - y1
    # bilinear interpolation
    P = Q11 * (1 - dx) * (1 - dy) + Q21 * dx * (1 - dy) + Q12 * (1 - dx) * dy + Q22 * dx * dy 
    # # save result
    output_bc_ptrs = output_ptr + batch*output_stride_b + channel*output_stride_c + offset_h*output_stride_h + offset_w
    tl.store(output_bc_ptrs, P, mask=mask)


def upsample_bilinear2d(
        input: torch.Tensor, 
        output_size: Optional[List[int]] = None,
        align_corners : bool = True, 
        scale_factors : Optional[List[float]] = None
) -> torch.Tensor:
    assert isinstance(input, torch.Tensor), "Input must be a PyTorch Tensor."
    assert input.dim() == 4, f"Input must be 4D tensor (got {input.dim()}D) - (N, C, H, W)."
    if output_size is not None:
        assert len(output_size) == 2, f"size must have 2 elements (H, W), got {len(output_size)}."
        assert all(isinstance(d, int) and d > 0 for d in output_size), f"Each element in size must be a positive integer, got {output_size}"
    if scale_factors is not None:
        assert all(isinstance(d, float) and d > 0 for d in scale_factors), f"Each element in size must be a positive float, got {scale_factors}"
    if output_size is not None and scale_factors is not None:
        raise ValueError("Only one of 'size' or 'scale_factor' can be provided.")
    if output_size is None and scale_factors is None:
        raise ValueError("Either 'size' or 'scale_factor' must be provided")
    N, C, H_in, W_in = input.shape
    if output_size is not None:
        H_out, W_out = output_size
    else:
        H_out = int(H_in * scale_factors[0])
        W_out = int(W_in * scale_factors[1])
    H_out, W_out = output_size
   
    output = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=input.device)
    # grid = lambda META: (N * C, triton.cdiv(H_out, META['BLOCK_SIZE_H']), triton.cdiv(W_out, META['BLOCK_SIZE_W']))    
    grid = lambda META: (N * C * triton.cdiv(W_out, META['BLOCK_SIZE_W']) * triton.cdiv(H_out, META['BLOCK_SIZE_H']),)
    upsample_bilinear2d_kernel[grid](
        input, output, 
        H_in, W_in, 
        H_out, W_out, 
        N, C,
        align_corners,
        *input.stride(), 
        *output.stride(),
    )
    return output

