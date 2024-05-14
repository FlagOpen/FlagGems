import torch
import triton
import triton.language as tl
import math
from .__libentry__ import libentry


@libentry()
@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0)
    mid_value = tl.cumprod(inp_val, axis=0)
    mid_ptr = mid + offset
    tl.store(mid_ptr, mid_value.to(inp_val.dtype))


@libentry()
@triton.jit
def prod_kernel_result(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr, block_size):
    offset = tl.arange(0, BLOCK_MID) * block_size
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE * block_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=1.0)
    # reult = mid_val * mid_val
    prod_val = tl.cumprod(mid_val, axis=0)
    out_ptr = out + tl.arange(0, BLOCK_MID)
    tl.store(out_ptr, prod_val.to(mid_val.dtype))


def prod(inp, dim=None, *, dtype=None):
    if __debug__:
        print("GEMS prod")
    if dtype is None:
        dtype = inp.dtype
    if dim is None:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid = torch.empty((mid_size, block_size), dtype=dtype, device=inp.device)
        out = torch.empty((mid_size,), dtype=dtype, device=inp.device)

        prod_kernel_mid[(mid_size, 1, 1)](inp, mid, M, block_size)
        new_mid = mid[:, -1]
        prod_kernel_result[(1, 1, 1)](new_mid, out, mid_size, block_mid, block_size)
        return out[-1]
    else:
        return inp
