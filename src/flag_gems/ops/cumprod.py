import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from ..utils import triton_lang_extension as tle

@libentry()
@triton.jit(do_not_specialize=["B", "C", "part_num"])
def scan_part_prod_kernel(
    inp,           
    out,           
    partial_prod,  
    B,             
    C,          
    part_num, 
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_c = tl.program_id(2)

    offset = pid_p * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < B

    base = pid_a * B * C + pid_c
    inp_ptrs = inp + base + offset * C
    out_ptrs = out + base + offset * C

    inp_vals = tl.load(inp_ptrs, mask=mask)

    if (tl.constexpr(inp_vals.dtype.is_int64())
        or tl.constexpr(inp_vals.dtype.is_uint64())
    ) or tl.constexpr(inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)

    val = tl.where(mask, inp_vals, 1)
    curr_prod = tl.cumprod(val, axis=0)

    num_valid = tl.sum(mask)
    last_value = tl.where(tl.arange(0, BLOCK_SIZE) == (num_valid - 1), curr_prod, 0.0)
    part_prod = tl.sum(last_value, axis=0)  
    partial_prod_ptr = partial_prod + pid_a * part_num * C + pid_c * part_num + pid_p
    tl.store(partial_prod_ptr, part_prod)
    tl.store(out_ptrs, curr_prod, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["B", "C", "part_num"])
def add_base_prod_kernel(
    out,
    partial_prod,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_c = tl.program_id(2)

    offset = pid_p * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < B

    base = pid_a * B * C + pid_c
    out_ptrs = out + base + offset * C
    out_vals = tl.load(out_ptrs, mask=mask)
    # Partial product pointers
    if pid_p > 0:
        partial_prod_ptrs = partial_prod + pid_a * part_num * C + pid_c * part_num + (pid_p - 1)
        last_part_prod = tl.load(partial_prod_ptrs)

        final_vals = out_vals * last_part_prod
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)


def scan_then_prod_1d(inp, out, n_ele, dtype):
    BLOCK_SIZE = 1024
    if n_ele <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(n_ele)
    part_num = math.ceil(n_ele / BLOCK_SIZE)
    partial_prod = torch.empty(part_num, dtype=dtype, device=inp.device)

    A = 1
    C = 1
    B = n_ele
    grid = (A, part_num, C)
    with torch.cuda.device(inp.device):
        scan_part_prod_kernel[grid](inp, out, partial_prod, B, C, part_num, BLOCK_SIZE)

    if part_num >= 2:
        scan_then_prod_1d(partial_prod, partial_prod, part_num, dtype)
        with torch.cuda.device(inp.device):
            add_base_prod_kernel[grid](out, partial_prod, B, C, part_num, BLOCK_SIZE)


def scan_then_prod_nd(inp, out, A, B, C, dtype):
    BLOCK_SIZE = 1024
    if B <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    partial_prod = torch.empty((A, part_num, C), dtype=dtype, device=inp.device)

    grid = (A, part_num, C)
    with torch.cuda.device(inp.device):
        scan_part_prod_kernel[grid](inp, out, partial_prod, B, C, part_num, BLOCK_SIZE)

    if part_num >= 2:
        scan_then_prod_nd(partial_prod, partial_prod, A, part_num, C, dtype)
        with torch.cuda.device(inp.device):
            add_base_prod_kernel[grid](out, partial_prod, B, C, part_num, BLOCK_SIZE)


def cumprod(inp, dim=1, *, dtype=None):
    logging.debug("GEMS CUMPROD")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim

    # Hanle the case where the input tensor is empty
    if 0 in shape:
        if dtype is None:
            dtype = inp.dtype
            if dtype is torch.bool:
                dtype = torch.int64
        return torch.empty_like(inp, dtype=dtype)

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

    compute_dtype = out.dtype
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if M == 1 and K == 1:
        scan_then_prod_1d(inp, out, N, compute_dtype)
    else:
        scan_then_prod_nd(inp, out, M, N, K, compute_dtype)
    return out