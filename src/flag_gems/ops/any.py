import torch
import triton
import math
import triton.language as tl
from ..utils import libentry
import logging


# torch.any: Tests if any elements in input evaluate to True.
#            If the dtype of input is not BOOL, then test if any elements in input evaluate to non-zero value
# In triton function, test if any elements in input evaluate to non-zero value is ok. 
def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@triton.jit
def reduce_any(a, b):
    return tl.maximum(a, b)


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def any_kernel_dim(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _any = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0.0)
        _any = _any or (a != 0)
    any = tl.reduce(_any, axis=1, combine_fn=reduce_any)
    tl.store(out, any[:, None], row_mask)


@libentry()
@triton.jit
def any_kernel_1(
    inp,
    mid,
    n_elements,
    mid_size, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    any_val = tl.max(inp_val != 0, axis=0)
    mid_ptr = mid + pid
    mid_mask = pid < mid_size
    tl.store(mid_ptr, any_val, mask=mid_mask)


@libentry()
@triton.jit
def any_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0)
    any_val = tl.max(mid_val, axis=0)
    tl.store(out, any_val)


def any(inp):
    logging.debug("GEMS any")
    n_elements = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    dtype = inp.dtype

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    any_kernel_1[(mid_size, 1)](inp, mid, n_elements, mid_size, block_size)
    any_kernel_2[(1, 1)](mid, out, mid_size, block_mid)

    return out


def any_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS any_dim")
    shape = list(inp.shape)
    if dim is None:
        dim = list(range(inp.ndim))
    else:
        assert (dim >= -inp.ndim and dim < inp.ndim) , "Invalid dim" 
        dim = [dim % inp.ndim]
    order = [i for i in range(inp.ndim) if i not in dim] + dim
    inp = inp.permute(order).contiguous()
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
    )
    any_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def any_dims(inp, dim=None, keepdim=False):
    logging.debug("GEMS any_dims")
    if dim is None or isinstance(dim, int):
        return any_dim(inp, dim=dim, keepdim=keepdim)
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim" 

    shape = list(inp.shape)
    dim = sorted([d % inp.ndim for d in dim])
    order = [i for i in range(inp.ndim) if i not in dim] + dim
    inp = inp.permute(order).contiguous()
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=torch.bool, device=inp.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
    )
    any_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out