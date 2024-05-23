import torch
import triton
import math
import triton.language as tl
from ..utils import libentry
import logging


# torch.all: Tests if all elements in input evaluate to True.
#            If the dtype of input is not BOOL, then test if all elements in input evaluate to non-zero value
# In triton function, test if all elements in input evaluate to non-zero value is ok. 
def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def all_kernel_dim(
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

    all = tl.full([BLOCK_M, 1], value=1., dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=1.0)
        min_val = tl.min(tl.abs(a), axis=1)[:, None]
        all = tl.minimum(all, min_val)
    tl.store(out, all, row_mask)


@libentry()
@triton.jit
def all_kernel_1(
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
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0)
    all_val = tl.min(tl.abs(inp_val), axis=0)
    mid_ptr = mid + pid
    mid_mask = mid_ptr < mid_size
    tl.store(mid_ptr, all_val, mask=mid_mask)


@libentry()
@triton.jit
def all_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=1.0)
    all_val = tl.min(mid_val, axis=0)
    tl.store(out, all_val)


def all(inp):
    logging.debug("GEMS all")
    n_elements = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    all_kernel_1[(mid_size, 1)](inp, mid, n_elements, mid_size, block_size)
    all_kernel_2[(1, 1)](mid, out, mid_size, block_mid)

    return out


def all_dim(inp, dim=None, keepdim=False): 
    logging.debug("GEMS all_dim")
    if dim is None:
        dim = list(range(inp.ndim))
    else:
        dim = [dim]
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim" 
    dtype = inp.dtype

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
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
    all_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
    

def all_dims(inp, dim=None, keepdim=False):
    logging.debug("GEMS all_dims")
    if dim is None:
        dim = list(range(inp.ndim))
    if isinstance(dim, int):
        dim = [dim]
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim" 
    dtype = inp.dtype

    shape = list(inp.shape)
    dim = sorted([d % inp.ndim for d in dim])
    order = [i for i in range(inp.ndim) if i not in dim] + dim
    if order == shape:
        inp = inp.contiguous()
    else:
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
    all_kernel_dim[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out