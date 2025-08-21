import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

@triton.jit
def diff_kernel_flat(
    input_ptr, output_ptr,
    B,
    L,
    out_L,
    input_stride,
    output_stride,
    N: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs  = start + tl.arange(0, BLOCK)
    total = B * out_L

    mask = offs < total
    batch = offs // out_L
    pos   = offs % out_L

    idx0 = batch * input_stride + pos
    idx1 = idx0 + N

    x0 = tl.load(input_ptr + idx0, mask=mask)
    x1 = tl.load(input_ptr + idx1, mask=mask)
    res = x1 - x0

    out_idx = batch * output_stride + pos
    tl.store(output_ptr + out_idx, res, mask=mask)


@triton.jit
def diff_kernel_1d_vectorized(
    input_ptr, output_ptr,
    N, out_L,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_WIDTH: tl.constexpr
):
    """1D向量化内核"""
    pid = tl.program_id(0)
    
    base_offset = pid * BLOCK_SIZE
    
    for vec_start in range(0, BLOCK_SIZE, VECTOR_WIDTH):
        offsets = base_offset + vec_start + tl.arange(0, VECTOR_WIDTH)
        mask = offsets < out_L
        
        x0_vec = tl.load(input_ptr + offsets, mask=mask, other=0.0, cache_modifier=".ca")
        x1_vec = tl.load(input_ptr + offsets + N, mask=mask, other=0.0, cache_modifier=".ca")
        
        result = x1_vec - x0_vec
        tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def diff_kernel_2d_tiled(
    input_ptr, output_ptr,
    B, L, out_L,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    N: tl.constexpr,
    dim: tl.constexpr,
    TILE_B: tl.constexpr,
    TILE_L: tl.constexpr
):
    """2D tile化内核，优化缓存使用"""
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    b_start = pid_b * TILE_B
    l_start = pid_l * TILE_L
    
    b_offsets = b_start + tl.arange(0, TILE_B)
    l_offsets = l_start + tl.arange(0, TILE_L)
    
    if dim == 0:
        b_mask = b_offsets < (B - N)
        l_mask = l_offsets < L
        mask = b_mask[:, None] & l_mask[None, :]
        
        addr_base = b_offsets[:, None] * input_stride_0 + l_offsets[None, :] * input_stride_1
        addr_next = (b_offsets[:, None] + N) * input_stride_0 + l_offsets[None, :] * input_stride_1
        
        x0 = tl.load(input_ptr + addr_base, mask=mask, other=0.0)
        x1 = tl.load(input_ptr + addr_next, mask=mask, other=0.0)
        
        out_addr = b_offsets[:, None] * output_stride_0 + l_offsets[None, :] * output_stride_1
        
    else:
        b_mask = b_offsets < B
        l_mask = l_offsets < out_L
        mask = b_mask[:, None] & l_mask[None, :]
        
        addr_base = b_offsets[:, None] * input_stride_0 + l_offsets[None, :] * input_stride_1
        addr_next = b_offsets[:, None] * input_stride_0 + (l_offsets[None, :] + N) * input_stride_1
        
        x0 = tl.load(input_ptr + addr_base, mask=mask, other=0.0)
        x1 = tl.load(input_ptr + addr_next, mask=mask, other=0.0)
        
        out_addr = b_offsets[:, None] * output_stride_0 + l_offsets[None, :] * output_stride_1
    
    result = x1 - x0
    tl.store(output_ptr + out_addr, result, mask=mask)


@triton.jit
def diff_kernel_3d_optimized(
    input_ptr, output_ptr,
    shape_0, shape_1, shape_2,
    input_stride_0, input_stride_1, input_stride_2,
    output_stride_0, output_stride_1, output_stride_2,
    N: tl.constexpr,
    diff_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    base_idx = pid * BLOCK_SIZE
    local_offsets = tl.arange(0, BLOCK_SIZE)
    global_offsets = base_idx + local_offsets
    
    out_shape_0 = shape_0 - N if diff_dim == 0 else shape_0
    out_shape_1 = shape_1 - N if diff_dim == 1 else shape_1
    out_shape_2 = shape_2 - N if diff_dim == 2 else shape_2
    total_output_elements = out_shape_0 * out_shape_1 * out_shape_2
    
    mask = global_offsets < total_output_elements
    
    idx_2 = global_offsets % out_shape_2
    temp = global_offsets // out_shape_2
    idx_1 = temp % out_shape_1
    idx_0 = temp // out_shape_1

    valid_0 = idx_0 < out_shape_0
    valid_1 = idx_1 < out_shape_1
    valid_2 = idx_2 < out_shape_2
    
    if diff_dim == 0:
        input_valid = (idx_0 + N) < shape_0
        in_addr_0 = idx_0 * input_stride_0 + idx_1 * input_stride_1 + idx_2 * input_stride_2
        in_addr_1 = (idx_0 + N) * input_stride_0 + idx_1 * input_stride_1 + idx_2 * input_stride_2
    elif diff_dim == 1:
        input_valid = (idx_1 + N) < shape_1
        in_addr_0 = idx_0 * input_stride_0 + idx_1 * input_stride_1 + idx_2 * input_stride_2
        in_addr_1 = idx_0 * input_stride_0 + (idx_1 + N) * input_stride_1 + idx_2 * input_stride_2
    else:
        input_valid = (idx_2 + N) < shape_2
        in_addr_0 = idx_0 * input_stride_0 + idx_1 * input_stride_1 + idx_2 * input_stride_2
        in_addr_1 = idx_0 * input_stride_0 + idx_1 * input_stride_1 + (idx_2 + N) * input_stride_2
    
    final_mask = mask & valid_0 & valid_1 & valid_2 & input_valid

    x0 = tl.load(input_ptr + in_addr_0, mask=final_mask, other=0.0)
    x1 = tl.load(input_ptr + in_addr_1, mask=final_mask, other=0.0)
    result = x1 - x0
    

    out_addr = idx_0 * output_stride_0 + idx_1 * output_stride_1 + idx_2 * output_stride_2
    tl.store(output_ptr + out_addr, result, mask=final_mask)


def _diff_once(
    input: Tensor,
    # n: int = 1,
    dim: int = -1,
    # prepend: Optional[Tensor] = None,
    # append: Optional[Tensor] = None
) -> Tensor:
    # if n <= 0:
    #     return input.clone()

    # # 拼接处理
    # if prepend is not None:
    #     input = torch.cat([prepend, input], dim=dim)
    # if append is not None:
    #     input = torch.cat([input, append], dim=dim)

    input = input.contiguous()
    dim = dim % input.ndim
    shape = list(input.shape)
    L = shape[dim]
    # if n >= L:
    #     shape[dim] = 0
    #     return input.new_empty(shape)
    out_L = L - 1
    if out_L <= 0:
        shape[dim] = 0
        return input.new_empty(shape)
    shape[dim] = out_L
    output = input.new_empty(shape)

    if input.ndim == 1:
        total = out_L
        BLOCK = 1024
        num_blocks = triton.cdiv(total, BLOCK)
        diff_kernel_flat[(num_blocks,)](
            input, output,
            1, L, out_L,
            input.stride(0), output.stride(0),
            1, BLOCK=BLOCK
        )

    elif input.ndim == 2:
        B, L_orig = input.shape
        out_L = L_orig - 1
        total = B * out_L
        aspect = B / L_orig
        SMALL_THRESHOLD = 1 << 16
        if aspect > 40 or aspect < 0.025:
            flat_in = input.view(-1)
            flat_out = output.view(-1)
            if total < SMALL_THRESHOLD:
                BLOCK = 1024
                num_blocks = triton.cdiv(total, BLOCK)
                diff_kernel_flat[(num_blocks,)](
                    flat_in, flat_out,
                    1, L_orig * B, total,
                    0, 0,
                    1, BLOCK=BLOCK
                )
            else:
                BLOCK_SIZE = min(4096, max(256, triton.next_power_of_2(total // 16)))
                VECTOR_WIDTH = min(128, max(32, BLOCK_SIZE // 32))
                num_blocks = triton.cdiv(total, BLOCK_SIZE)
                diff_kernel_1d_vectorized[(num_blocks,)](
                    flat_in, flat_out,
                    1, total, BLOCK_SIZE, VECTOR_WIDTH,
                    num_warps=min(32, max(4, BLOCK_SIZE // 128))
                )
            return output
        
        if dim == 0:
            target_B, target_L = B - 1, L_orig
            TILE_B = min(128, max(16, triton.next_power_of_2(target_B // 8)))
            TILE_L = min(128, max(32, triton.next_power_of_2(target_L // 8)))
        else:
            target_B, target_L = B, out_L
            TILE_B = min(64, max(16, triton.next_power_of_2(target_B // 8)))
            TILE_L = min(256, max(32, triton.next_power_of_2(target_L // 8)))
        
        while TILE_B > 1 and target_B % TILE_B != 0 and TILE_B > target_B // 4:
            TILE_B //= 2
        while TILE_L > 1 and target_L % TILE_L != 0 and TILE_L > target_L // 4:
            TILE_L //= 2
        
        grid_B = triton.cdiv(target_B, TILE_B)
        grid_L = triton.cdiv(target_L, TILE_L)
        
        diff_kernel_2d_tiled[(grid_B, grid_L)](
            input, output, B, L_orig, out_L,
            input.stride(0), input.stride(1),
            output.stride(0), output.stride(1),
            1, dim, TILE_B, TILE_L,
            num_warps=min(16, max(4, (TILE_B * TILE_L) // 64))
        )


    elif input.ndim == 3:
        total = output.numel()
        BLOCK_SIZE = min(2048, max(512, triton.next_power_of_2(total // 512)))
        num_blocks = triton.cdiv(total, BLOCK_SIZE)
        diff_kernel_3d_optimized[(num_blocks,)](
            input, output,
            input.shape[0], input.shape[1], input.shape[2],
            input.stride(0), input.stride(1), input.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            1, dim, BLOCK_SIZE,
            num_warps=min(32, max(8, BLOCK_SIZE // 128))
        )
  
    return output


def diff(
    input: Tensor,
    n: int = 1,
    dim: int = -1,
    prepend: Optional[Tensor] = None,
    append: Optional[Tensor] = None
) -> Tensor:
    if n <= 0:
        return input.clone()

    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)
    
    out = input

    for _ in range(n):
        out = _diff_once(out, dim)
    return out
