import logging
import math

import torch
import triton
import triton.language as tl
import triton.language.core as core

from ..utils import libentry


@triton.jit
def _min_combine(a, idxa, b, idxb):
    if a < b:
        return a, idxa
    else:
        return b, idxb


@triton.jit
@core._add_scan_docstr("cummin")
def tl_cummin(input, index, axis=0):
    # input = (core._promote_bfloat16_to_float32(inp) for inp in input)
    return core.associative_scan((input, index), axis, _min_combine)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def add_base_min_kernel(
    out,
    partial_min,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)

    if pid > 0:
        partial_min_ptrs = partial_min + pid - 1
        last_part_min_via_min = tl.load(partial_min_ptrs)

        final_vals = tl.min(out_vals, last_part_min_via_min)
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "part_num"])
def scan_part_min_kernel(
    inp,
    out,
    out_indices,
    partial_min,
    partial_min_indices,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    result, indices = tl_cummin(inp_vals, offset, axis=0)

    part_min_via_min, part_min_indices_via_min = tl.min(
        inp_vals, axis=0, return_indices=True
    )

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    out_indices_ptrs = out_indices + offset
    indices_offset = pid * BLOCK_SIZE + indices
    in_indices_ptrs = out_indices + indices_offset
    indices_mask = indices_offset < n_elements
    true_out_indices = tl.load(in_indices_ptrs, mask=indices_mask)
    tl.store(out_indices_ptrs, true_out_indices, mask=mask)

    partial_min_ptrs = partial_min + pid
    tl.store(partial_min_ptrs, part_min_via_min)
    partial_min_indices_ptrs = partial_min_indices + pid
    tl.store(partial_min_indices_ptrs, part_min_indices_via_min)


def scan_then_fan_col(inp, out, out_indices, n_ele, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if n_ele <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(n_ele)
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    partial_min = torch.empty(part_num, dtype=dtype, device=inp.device)
    partial_min_indices = torch.empty(part_num, dtype=torch.int64, device=inp.device)

    grid = (part_num,)
    with torch.cuda.device(inp.device):
        scan_part_min_kernel[grid](
            inp, out, out_indices, partial_min, partial_min_indices, n_ele, BLOCK_SIZE
        )

    if part_num >= 2:
        scan_then_fan_col(
            partial_min, partial_min, partial_min_indices, part_num, dtype
        )
        with torch.cuda.device(inp.device):
            add_base_min_kernel[grid](out, partial_min, n_ele, BLOCK_SIZE)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def scan_part_min_abc_kernel(
    inp,
    out,
    partial_min,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    offset = a_idx * B * C + b_idx * C + c_idx
    base_part_offset = a_idx * part_num * C + c_idx
    part_offset = base_part_offset + pid_b * C

    mask = b_idx < B
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask)
    if (
        tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    result = tl_cummin(inp_vals, axis=0)

    part_min_via_min = tl.min(inp_vals)

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    partial_min_ptrs = partial_min + part_offset
    tl.store(partial_min_ptrs, part_min_via_min)


@libentry()
@triton.jit(do_not_specialize=["part_num"])
def add_base_min_abc_kernel(
    out,
    partial_min,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    base_offset = a_idx * B * C + c_idx
    offset = base_offset + b_idx * C
    base_part_offset = a_idx * part_num * C + c_idx
    last_part_offset = base_part_offset + (pid_b - 1) * C

    mask = b_idx < B
    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)

    if pid_b > 0:
        partial_min_ptrs = partial_min + last_part_offset
        last_part_min_via_min = tl.load(partial_min_ptrs)

        final_vals = tl.min(out_vals, last_part_min_via_min)
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)


def scan_then_fan(inp, out, out_indices, A, B, C, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if B <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    partial_min = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)

    grid = (A, part_num, C)
    with torch.cuda.device(inp.device):
        scan_part_min_abc_kernel[grid](
            inp, out, partial_min, B, C, part_num, BLOCK_SIZE
        )

    if part_num >= 2:
        scan_then_fan(partial_min, partial_min, A, part_num, C, dtype)
        with torch.cuda.device(inp.device):
            add_base_min_abc_kernel[grid](out, partial_min, B, C, part_num, BLOCK_SIZE)


def cummin(inp, dim=1, *, dtype=None):
    logging.debug("GEMS cummin")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64
    out = torch.empty_like(inp, dtype=dtype)
    # out_indices = torch.empty_like(inp, dtype=torch.int64)
    out_indices = torch.arange(N, device=inp.device).reshape(shape)

    compute_dtype = out.dtype
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if M == 1 and K == 1:
        logging.debug("scan then fan col")
        scan_then_fan_col(inp, out, out_indices, N, compute_dtype)
    else:
        scan_then_fan(inp, out, out_indices, M, N, K, compute_dtype)
    logging.debug(out)
    logging.debug(out_indices)
    return out, out_indices
