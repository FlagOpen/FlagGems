import logging
import math
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def argmin_kernel_1(
    inp,
    mid_val,
    mid_idx,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    vals = tl.load(inp + offset, mask=mask, other=max_value)
    local_min, local_argmin = tl.min(
        vals, axis=0, return_indices=True, return_indices_tie_break_left=True
    )
    local_argmin = local_argmin + pid * BLOCK_SIZE
    tl.store(mid_val + pid, local_min)
    tl.store(mid_idx + pid, local_argmin)


@libentry()
@triton.jit
def argmin_kernel_2(
    mid_val,
    mid_idx,
    out_idx,
    MID_SIZE,
    BLOCK_MID: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_MID)
    mask = offsets < MID_SIZE
    dtype = mid_val.type.element_ty
    max_value = get_dtype_max(dtype)
    vals = tl.load(mid_val + offsets, mask=mask, other=max_value)
    pos = tl.argmin(vals, axis=0)
    final_idx = tl.load(mid_idx + pos)
    tl.store(out_idx, final_idx)


def argmin(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS ARGMIN")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        mid_val = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_idx = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)
        with torch_device_fn.device(inp.device):
            argmin_kernel_1[(mid_size, 1, 1)](inp, mid_val, mid_idx, M, block_size)
            argmin_kernel_2[(1, 1, 1)](mid_val, mid_idx, out, mid_size, block_mid)
        return out
    else:
        return argmin_dim(inp, dim=dim, keepdim=keepdim)


def argmin_out(inp, *, out):
    logger.debug("GEMS ARGMIN_OUT")
    M = inp.numel()
    dtype = inp.dtype
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    mid_val = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    mid_idx = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device):
        argmin_kernel_1[(mid_size, 1, 1)](inp, mid_val, mid_idx, M, block_size)
        argmin_kernel_2[(1, 1, 1)](mid_val, mid_idx, out, mid_size, block_mid)
    return out


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def argmin_dim_kernel_non_inner(
    out_idx_ptr,
    in_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]
    dtype = in_ptr.type.element_ty
    max_value = get_dtype_max(dtype)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_off = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        tile = tl.load(in_ptr + inp_off, mask=mask, other=max_value)
        local_min, local_arg = tl.min(
            tile, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        local_arg = local_arg[None, :].to(tl.int64)
        out_off = pid_m * K + k_offsets
        tl.store(out_idx_ptr + out_off, local_arg, mask=(k_offsets < K))
    else:
        cur_min = tl.full([1, TILE_K], value=float("inf"), dtype=tl.float32)
        cur_arg = tl.full([1, TILE_K], value=0, dtype=tl.int64)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_off = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            tile = tl.load(in_ptr + inp_off, mask=mask, other=max_value)
            local_min, local_arg = tl.min(
                tile, axis=0, return_indices=True, return_indices_tie_break_left=True
            )
            local_min = local_min[None, :].to(tl.float32)
            glob_arg = (start_n + local_arg)[None, :]
            update = local_min < cur_min
            cur_min = tl.where(update, local_min, cur_min)
            cur_arg = tl.where(update, glob_arg.to(tl.int64), cur_arg)
        out_off = pid_m * K + k_offsets
        tl.store(out_idx_ptr + out_off, cur_arg, mask=(k_offsets < K))


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def argmin_dim_kernel_inner(
    out_idx_ptr,
    in_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)
    dtype = in_ptr.type.element_ty
    max_value = get_dtype_max(dtype)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_off = pid_m * N + n_offsets
        mask = n_offsets < N
        vec = tl.load(in_ptr + inp_off, mask=mask, other=max_value)
        vmin, vidx = tl.min(
            vec, axis=0, return_indices=True, return_indices_tie_break_left=True
        )
        tl.store(out_idx_ptr + pid_m, vidx.to(tl.int64))
    else:
        cur_min = tl.full((), value=float("inf"), dtype=tl.float32)
        cur_arg = tl.full((), value=0, dtype=tl.int64)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_off = pid_m * N + n_offsets
            mask = n_offsets < N
            vec = tl.load(in_ptr + inp_off, mask=mask, other=max_value)
            vmin, vidx = tl.min(
                vec, axis=0, return_indices=True, return_indices_tie_break_left=True
            )
            vmin = vmin.to(tl.float32)
            glob_arg = start_n + vidx
            update = vmin < cur_min
            cur_min = tl.where(update, vmin, cur_min)
            cur_arg = tl.where(update, glob_arg.to(tl.int64), cur_arg)
        tl.store(out_idx_ptr + pid_m, cur_arg)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def argmin_dim_kernel(
    inp_ptr,
    out_idx_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = pid < M
    inp_ptr = inp_ptr + pid * N
    dtype = inp_ptr.type.element_ty
    max_value = get_dtype_max(dtype)
    cur_min = tl.full([BLOCK_M, 1], value=float("inf"), dtype=tl.float32)
    cur_arg = tl.full([BLOCK_M, 1], value=0, dtype=tl.int64)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        tile = tl.load(inp_ptr + cols, mask=mask, other=max_value)
        vmin, vidx = tl.min(
            tile, axis=1, return_indices=True, return_indices_tie_break_left=True
        )
        vmin = vmin[:, None]
        vidx = vidx[:, None]
        glob_arg = off + vidx
        update = vmin < cur_min
        cur_min = tl.where(update, vmin, cur_min)
        cur_arg = tl.where(update, glob_arg.to(tl.int64), cur_arg)
    out_ptr = out_idx_ptr + tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(out_ptr, cur_arg[:, 0], mask=row_mask[:, 0])


def argmin_dim_comm(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS ARGMIN_DIM")
    assert dim is not None
    if isinstance(dim, (list, tuple)):
        assert len(dim) == 1
        dim = dim[0]
    dim = int(dim)
    assert -inp.ndim <= dim < inp.ndim
    dim = dim % inp.ndim
    shape = list(inp.shape)
    N = inp.shape[dim]
    M = reduce(lambda x, y: x * y, shape[:dim], 1)
    inp = inp.contiguous()
    K = inp.numel() // M // N
    out_shape = shape[:]
    out_shape[dim] = 1
    if out is None:
        out = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
    with torch_device_fn.device(inp.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            argmin_dim_kernel_non_inner[grid](out, inp, M, N, K)
        else:
            grid = (M, 1, 1)
            argmin_dim_kernel_inner[grid](out, inp, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def argmin_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS ARGMIN_DIM (wrapper)")
    if dim is None or dim == []:
        return argmin(inp, dim=None, keepdim=keepdim)
    return argmin_dim_comm(inp, dim=dim, keepdim=keepdim)
