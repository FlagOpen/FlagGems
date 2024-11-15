import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 32}, num_warps=8),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    offset_index = m_offset * K + pid_k
    # set mask
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    result_value, result_index = tl.max(inp_vals, axis=1, return_indices=True)

    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)

@libentry()
@triton.jit
def max_kernel_1_uncontiguous_2dim(
    inp_ptr,
    inp_stride0,
    inp_stride1,
    inp_shape0,
    inp_shape1,
    mid_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    block_start_offset = inp_ptr + inp_stride0 * block_idx
    max_val = -float("inf")
    for offset in range(0, inp_shape1, BLOCK_SIZE):
        process_data_offset = block_start_offset + (offset + tl.arange(0, BLOCK_SIZE)) * inp_stride1
        mask = (offset + tl.arange(0,BLOCK_SIZE))<inp_shape1
        inp_vals = tl.load(process_data_offset , mask=mask, other=-float("inf"))
        max_val_sub = tl.max(inp_vals)
        max_val = tl.maximum(max_val, max_val_sub)
    tl.store(mid_ptr + block_idx, max_val)

@triton.jit
def max_kernel_1_uncontiguous_2dim_v2(
    inp_ptr,
    inp_stride0,inp_stride1,
    inp_shape0,inp_shape1,
    mid_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    block_nums = tl.num_programs(0)

    number_of_row_to_process = tl.cdiv(inp_shape0, block_nums)
    start_row = block_idx * number_of_row_to_process
    for row_index in range(number_of_row_to_process):
        row_to_process = start_row + row_index
        if row_to_process < inp_shape0:
            row_start_offset = inp_ptr + inp_stride0 * row_to_process
            max_val = -float("inf")
            for offset in range(0, inp_shape1, BLOCK_SIZE):
                process_data_offset = row_start_offset + (offset + tl.arange(0, BLOCK_SIZE)) * inp_stride1
                mask = (offset + tl.arange(0,BLOCK_SIZE))<inp_shape1
                inp_vals = tl.load(process_data_offset, mask=mask, other=-float("inf"))
                max_val_sub = tl.max(inp_vals)
                max_val = tl.maximum(max_val, max_val_sub)
            tl.store(mid_ptr + row_to_process, max_val)


def max_2d_uncontiguous(inp):
    if inp.shape[0]<= 4096:
        # use one block to process one row, and the number of blocks is equal to the number os rows
        out = torch.empty([], dtype=inp.dtype, device=inp.device)
        mid = torch.empty((inp.shape[0],), dtype=inp.dtype, device=inp.device)
        max_kernel_1_uncontiguous_2dim[(inp.shape[0],)](inp,
                                                        inp.stride(0),inp.stride(1),
                                                        inp.shape[0],inp.shape[1],
                                                        mid,
                                                        256)
        max_kernel_2[(1, 1, 1)](mid, out, inp.shape[0], 256)
        return out
    
    if inp.shape[0]>4096:
        # use one block to process multiple rows
        out = torch.empty([], dtype=inp.dtype, device=inp.device)
        mid = torch.ones((inp.shape[0],), dtype=inp.dtype, device=inp.device)
        max_kernel_1_uncontiguous_2dim_v2[(1024,)](inp,
                                                inp.stride(0),inp.stride(1),
                                                inp.shape[0],inp.shape[1],
                                                mid,
                                                2)
        max_kernel_2[(1, 1, 1)](mid, out, inp.shape[0], triton.next_power_of_2(inp.shape[0]))
        return out
    
def max(inp: torch.Tensor):
    logging.debug("GEMS MAX")
    if inp.ndim == 2 and inp.is_contiguous() == False:
        out = max_2d_uncontiguous(inp)
        return out
    inp = inp.contiguous()
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch.cuda.device(inp.device):
        max_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def max_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.cuda.device(inp.device):
        max_kernel[grid](inp, out_value, out_index, M, N, K)
    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value, indices=out_index)
    return out
