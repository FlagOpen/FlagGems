# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_min

from ..utils.tle_copy import tle_copy

logger = logging.getLogger(__name__)


_FULL_REDUCTION_BLOCK_SIZE = 8192
_FAST_MAX_N = 64
_FAST_BN_FP16 = (1024, 256, 512, 128, 64)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64)
_FAST_BM_FP16 = (128, 64, 32, 256, 512, 16, 8, 4, 2)
_FAST_BM_FP32 = (64, 128, 32, 256, 512, 16, 8, 4, 2)
_VIEW_BN_FP16 = (512, 256, 128, 64, 1024)
_VIEW_BN_FP32 = (512, 256, 128, 64, 1024)
_VIEW_BM_FP16 = (128, 64, 32, 256, 512, 16, 8, 4, 2)
_VIEW_BM_FP32 = (64, 128, 32, 256, 512, 16, 8, 4, 2)
_MAX_FAST_TILE_LANES = 8192
_NEW_MAX_VR = 65536

# N above this width cannot be reduced (with return_indices) in a single XPU
# tile for integers; handled by the two-stage reduction below.
MAX_TILE_N = 8192
STAGE_BLOCK_N = 2048
STAGE_BLOCK_M = 32


def _is_fast_dtype(dtype):
    return dtype in (torch.float16, torch.float32, torch.bfloat16)


def _pick_chunk_tile(VR, CHUNK, is_fp32):
    bns = _VIEW_BN_FP32 if is_fp32 else _VIEW_BN_FP16
    bms = _VIEW_BM_FP32 if is_fp32 else _VIEW_BM_FP16
    for min_trips in (2, 1):
        for bn in bns:
            if CHUNK % bn != 0 or CHUNK // bn < min_trips:
                continue
            for bm in bms:
                if VR % bm == 0 and bm * bn >= _MAX_FAST_TILE_LANES:
                    return bm, bn
    return None


def _pick_fast_tile(M, N, is_fp32):
    if N < _FAST_MAX_N:
        return None
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    for bn in bns:
        if N % bn != 0:
            continue
        for bm in bms:
            if M % bm != 0:
                continue
            if bm * bn >= _MAX_FAST_TILE_LANES:
                return bm, bn
    return None


@libentry()
@triton.jit
def max_split_kernel(
    inp,
    part_val,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows.to(tl.int64) * N
    row_mask = rows < M
    ic = 0
    for off in range(0, N, BLOCK_CHUNK):
        cols = off + tl.arange(0, BLOCK_CHUNK)[None, :]
        if NEED_MASK:
            mask = row_mask and cols < N
            a = tl.load(inp + cols, mask=mask, other=float("-inf")).to(tl.float32)
        else:
            a = tl.load(inp + cols).to(tl.float32)
        blk = tl.max(a, axis=1)[:, None]
        if NEED_MASK:
            tl.store(part_val + rows.to(tl.int64) + ic * M, blk, mask=row_mask)
        else:
            tl.store(part_val + rows.to(tl.int64) + ic * M, blk)
        ic += 1


@libentry()
@triton.jit
def max_chunk_kernel_col(
    part_val,
    best_c,
    M,
    NC,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c_off = tl.arange(0, BLOCK_C)
    cmask = c_off < NC
    vals = tl.load(
        part_val + rows[:, None] + c_off[None, :].to(tl.int64) * M,
        mask=row_mask[:, None] and cmask[None, :],
        other=float("-inf"),
    )
    _, bc = tl.max(
        vals, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    tl.store(best_c + rows.to(tl.int64), bc, mask=row_mask)


@libentry()
@triton.jit
def max_scan_kernel_col(
    inp,
    part_val,
    best_c,
    out_val,
    out_idx,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c = tl.load(best_c + rows)
    base = rows.to(tl.int64) * N + c.to(tl.int64) * BLOCK_CHUNK
    cols = tl.arange(0, BLOCK_CHUNK)
    col_ok = cols[None, :] < (N - c * BLOCK_CHUNK)[:, None]
    a = tl.load(
        inp + base[:, None] + cols[None, :],
        mask=row_mask[:, None] and col_ok,
        other=float("-inf"),
    ).to(tl.float32)
    u = a.to(tl.uint32, bitcast=True)
    key = tl.where(u < 0x80000000, u ^ 0x80000000, u ^ 0xFFFFFFFF)
    pack = (key.to(tl.int64) << 30) | (
        (BLOCK_CHUNK - 1 - cols).to(tl.int64) & 0x3FFFFFFF
    )
    blk = tl.max(pack, axis=1)
    pos = (BLOCK_CHUNK - 1) - (blk & 0x3FFFFFFF) + c.to(tl.int64) * BLOCK_CHUNK
    m = tl.load(part_val + rows.to(tl.int64) + c.to(tl.int64) * M, mask=row_mask)
    tl.store(out_val + rows.to(tl.int64), m, mask=row_mask)
    tl.store(out_idx + rows.to(tl.int64), pos, mask=row_mask)


@libentry()
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offset < M
        min_value = get_dtype_min(inp.type.element_ty)
        inp_val = tl.load(inp + offset, mask=mask, other=min_value)
    else:
        inp_val = tl.load(inp + offset)
    tl.store(mid + pid, tl.max(inp_val))


def heur_m_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))


def heur_n_block_size(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)
    max_values = tl.full([BLOCK_M], dtype=acc_type, value=min_value)
    argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_vals = tl.load(inp + offset, mask=mask, other=min_value)
        local_max, local_argmax = tl.max(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_value_ptrs, max_values, mask=mask1)
    tl.store(out_index_ptrs, argmax_values, mask=mask1)


@libentry()
@triton.jit
def max_stage1(
    inp,
    part_val,
    part_idx,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_c = ext.program_id(1)
    pid_k = ext.program_id(2)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    chunk_start = pid_c * BLOCK_N
    n_offset = chunk_start + tl.arange(0, BLOCK_N)
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    vals = tl.load(inp + offset, mask=mask, other=min_value)
    lmax, largmax = tl.max(
        vals, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    gidx = chunk_start + largmax
    part_offset = m_offset * NUM_CHUNKS * K + pid_c * K + pid_k
    pmask = m_offset < M
    tl.store(part_val + part_offset, lmax, mask=pmask)
    tl.store(part_idx + part_offset, gidx, mask=pmask)


@libentry()
@triton.jit
def max_stage2(
    part_val,
    part_idx,
    out_value,
    out_index,
    M,
    K: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_offset = tl.arange(0, BLOCK_C)
    dtype = part_val.type.element_ty
    min_value = get_dtype_min(dtype)
    offset = m_offset[:, None] * NUM_CHUNKS * K + c_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and c_offset[None, :] < NUM_CHUNKS
    vals = tl.load(part_val + offset, mask=mask, other=min_value)
    best_val, best_c = tl.max(
        vals, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    gather = m_offset * NUM_CHUNKS * K + best_c * K + pid_k
    pmask = m_offset < M
    res_idx = tl.load(part_idx + gather, mask=pmask)
    tl.store(out_value + m_offset * K + pid_k, best_val, mask=pmask)
    tl.store(out_index + m_offset * K + pid_k, res_idx, mask=pmask)


@libentry()
@triton.jit
def max_kernel_2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows.to(tl.int64) * N
    out = out + rows
    row_mask = rows < M

    acc = tl.full([BLOCK_M, 1], value=float("-inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            col_mask = cols < N
            mask = row_mask and col_mask
            a = tl.load(inp + cols, mask=mask, other=float("-inf")).to(tl.float32)
            a = tl.where(mask, a, float("-inf"))
            blk = tl.max(a, axis=1)[:, None]
        else:
            a = tl.load(inp + cols).to(tl.float32)
            blk = tl.max(a, axis=1)[:, None]
        acc = tl.maximum(acc, blk)
    if NEED_MASK:
        tl.store(out, acc, row_mask)
    else:
        tl.store(out, acc)


@libentry()
@triton.jit
def max_chunk_val_kernel(
    inp,
    part_val,
    M,
    W,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_c = tl.where(rows < M, rows, M - 1)
    inp = inp + rows_c.to(tl.int64) * W
    out = part_val + rows.to(tl.int64)
    row_mask = rows < M
    acc = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32)
    for off in range(0, W, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        a = tl.load(inp + cols).to(tl.float32)
        acc = tl.maximum(acc, a)
    tl.store(out, tl.max(acc, axis=1)[:, None], row_mask)


@libentry()
@triton.jit
def max_chunk_kernel(
    part_val,
    best_c,
    M,
    NC,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c_off = tl.arange(0, BLOCK_C)
    cmask = c_off < NC
    vals = tl.load(
        part_val + rows[:, None].to(tl.int64) * NC + c_off[None, :],
        mask=row_mask[:, None] and cmask[None, :],
        other=float("-inf"),
    )
    _, bc = tl.max(
        vals, axis=1, return_indices=True, return_indices_tie_break_left=True
    )
    tl.store(best_c + rows.to(tl.int64), bc, mask=row_mask)


@libentry()
@triton.jit
def max_scan_kernel(
    inp,
    part_val,
    best_c,
    out_val,
    out_idx,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c = tl.load(best_c + rows)
    base = rows.to(tl.int64) * N + c.to(tl.int64) * BLOCK_CHUNK
    cols = tl.arange(0, BLOCK_CHUNK)
    col_ok = cols[None, :] < (N - c * BLOCK_CHUNK)[:, None]
    a = tl.load(
        inp + base[:, None] + cols[None, :],
        mask=row_mask[:, None] and col_ok,
        other=float("-inf"),
    ).to(tl.float32)
    u = a.to(tl.uint32, bitcast=True)
    key = tl.where(u < 0x80000000, u ^ 0x80000000, u ^ 0xFFFFFFFF)
    pack = (key.to(tl.int64) << 30) | (
        (BLOCK_CHUNK - 1 - cols).to(tl.int64) & 0x3FFFFFFF
    )
    blk = tl.max(pack, axis=1)
    pos = (BLOCK_CHUNK - 1) - (blk & 0x3FFFFFFF) + c.to(tl.int64) * BLOCK_CHUNK
    m = tl.load(
        part_val + rows.to(tl.int64) * (N // BLOCK_CHUNK) + c.to(tl.int64),
        mask=row_mask,
    )
    tl.store(out_val + rows.to(tl.int64), m, mask=row_mask)
    tl.store(out_idx + rows.to(tl.int64), pos, mask=row_mask)


def _pad_value(dtype):
    """Identity value for the max reduction: -inf for floats, dtype min otherwise."""
    if dtype.is_floating_point:
        return float("-inf")
    return torch.iinfo(dtype).min


def _pad_buffer(src, pad_to, device):
    n = src.numel()
    buf = torch.full((pad_to,), _pad_value(src.dtype), dtype=src.dtype, device=device)
    if n:
        if not tle_copy(src, buf[:n]):
            torch.ops.aten._copy_from(src, buf[:n], False)
    return buf


def _reduce_to_scalar(src, out):
    block = _FULL_REDUCTION_BLOCK_SIZE
    n = src.numel()
    data = src
    while n > block:
        n_pad = triton.cdiv(n, block) * block
        if n_pad > n:
            data = _pad_buffer(data, n_pad, data.device)
        mid_size = triton.cdiv(n, block)
        mid = torch.empty((mid_size,), dtype=data.dtype, device=data.device)
        max_kernel_1[(mid_size, 1)](data, mid, n, block, False, buffer_size_limit=2048)
        data = mid
        n = mid_size
    if n < block:
        data = _pad_buffer(data, block, data.device)
    max_kernel_1[(1, 1)](data, out, n, block, False, buffer_size_limit=2048)


def _max_flat(inp, out, device):
    """Full (dim=None) float reduction via mask-free 2-D row tiles plus a staged
    scalar reduce; every tail is padded so no masked/OOB load can occur."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    numel = inp.numel()
    pad_val = _pad_value(inp.dtype)
    if numel <= block:
        buf = torch.full((block,), pad_val, dtype=inp.dtype, device=device)
        if not tle_copy(inp, buf[:numel]):
            torch.ops.aten._copy_from(inp, buf[:numel], False)
        max_kernel_1[(1, 1)](buf, out, numel, block, False, buffer_size_limit=2048)
        return
    rows = numel // block
    res = numel - rows * block
    is_fp32 = inp.dtype == torch.float32
    bn = 1024 if not is_fp32 else 256
    if block % bn != 0:
        bn = next((b for b in _FAST_BN_FP32 if block % b == 0), block)
    bm = 128 if not is_fp32 else 64
    if rows % bm != 0:
        bm = next(
            (
                m
                for m in _FAST_BM_FP32
                if rows % m == 0 and m * bn >= _MAX_FAST_TILE_LANES
            ),
            128,
        )
    rows_exact = (rows // bm) * bm
    mid = torch.empty((rows + (1 if res else 0),), dtype=inp.dtype, device=device)
    with torch_device_fn.device(device):
        if rows_exact:
            max_kernel_2d[(rows_exact // bm, 1)](
                inp,
                mid,
                rows_exact,
                block,
                bm,
                bn,
                False,
                buffer_size_limit=2048,
            )
        if rows > rows_exact:
            tail_rows = rows - rows_exact
            tail_buf = torch.full(
                (bm * block,), pad_val, dtype=inp.dtype, device=device
            )
            src_tail = inp[rows_exact * block : rows * block]
            if not tle_copy(src_tail, tail_buf[: tail_rows * block]):
                torch.ops.aten._copy_from(
                    src_tail, tail_buf[: tail_rows * block], False
                )
            tail_mid = torch.empty((bm,), dtype=inp.dtype, device=device)
            max_kernel_2d[(1, 1)](
                tail_buf, tail_mid, bm, block, bm, bn, False, buffer_size_limit=2048
            )
            if not tle_copy(
                tail_mid[:tail_rows], mid[rows_exact : rows_exact + tail_rows]
            ):
                torch.ops.aten._copy_from(
                    tail_mid[:tail_rows],
                    mid[rows_exact : rows_exact + tail_rows],
                    False,
                )
        if res:
            res_buf = torch.full((block,), pad_val, dtype=inp.dtype, device=device)
            src_res = inp[rows * block :]
            if not tle_copy(src_res, res_buf[:res]):
                torch.ops.aten._copy_from(src_res, res_buf[:res], False)
            max_kernel_1[(1, 1)](
                res_buf, mid[rows:], res, block, False, buffer_size_limit=2048
            )
        _reduce_to_scalar(mid, out)


def _max_flat_int(inp, out, device):
    """Full (dim=None) reduction over a contiguous flat `inp` (integer)."""
    _reduce_to_scalar(inp, out)


def max(inp):
    logger.debug("GEMS_KUNLUNXIN MAX")
    dtype = inp.dtype
    inp = inp.contiguous().reshape(-1)
    out = torch.empty([], dtype=dtype, device=inp.device)
    if inp.numel() == 1:
        return inp.reshape([])
    with torch_device_fn.device(inp.device):
        if dtype.is_floating_point:
            _max_flat(inp, out, inp.device)
        else:
            _max_flat_int(inp, out, inp.device)
    return out


def max_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MAX_DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N
    dtype = inp.dtype

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if N == 1:
        out_index.zero_()
        with torch_device_fn.device(inp.device):
            if not tle_copy(inp, out_value):
                torch.ops.aten._copy_from(inp, out_value, False)
        if not keepdim:
            out_value = torch.squeeze(out_value, dim)
            out_index = torch.squeeze(out_index, dim)
        Max_out = namedtuple("max", ["values", "indices"])
        return Max_out(values=out_value, indices=out_index)

    if N >= _FAST_MAX_N and _is_fast_dtype(dtype):
        CHUNK = (
            1024
            if N % 1024 == 0
            else (
                512
                if N % 512 == 0
                else (
                    256
                    if N % 256 == 0
                    else (128 if N % 128 == 0 else 64 if N % 64 == 0 else 0)
                )
            )
        )
        M2 = M * K
        is_fp32 = dtype == torch.float32
        if CHUNK:
            nc = N // CHUNK
            VR = M2 * nc
            if M2 >= 8 and nc <= 1024 and VR >= _NEW_MAX_VR:
                tile = _pick_chunk_tile(VR, CHUNK, is_fp32)
                if tile is not None:
                    block_m, block_n = tile
                    perm = [d for d in range(inp.dim()) if d != dim] + [dim]
                    view = inp.permute(perm)
                    if view.is_contiguous():
                        src = view
                    else:
                        src = torch.empty(
                            list(view.shape), dtype=dtype, device=inp.device
                        )
                        with torch_device_fn.device(inp.device):
                            if not tle_copy(view, src):
                                torch.ops.aten._copy_from(view, src, False)
                    part_val = torch.empty(VR, dtype=torch.float32, device=inp.device)
                    best_c = torch.empty((M2,), dtype=torch.int32, device=inp.device)
                    out_flat = out_value.reshape(-1)
                    out_idx_flat = out_index.reshape(-1)
                    bm2 = (
                        32
                        if M2 % 32 == 0
                        else (
                            16
                            if M2 % 16 == 0
                            else (
                                8
                                if M2 % 8 == 0
                                else (4 if M2 % 4 == 0 else 2 if M2 % 2 == 0 else 1)
                            )
                        )
                    )
                    bm3 = (
                        8
                        if M2 % 8 == 0
                        else (4 if M2 % 4 == 0 else (2 if M2 % 2 == 0 else 1))
                    )
                    with torch_device_fn.device(inp.device):
                        max_chunk_val_kernel[(VR // block_m,)](
                            src,
                            part_val,
                            VR,
                            CHUNK,
                            block_m,
                            block_n,
                            buffer_size_limit=2048,
                        )
                        max_chunk_kernel[(M2 // bm2,)](
                            part_val,
                            best_c,
                            M2,
                            nc,
                            bm2,
                            triton.next_power_of_2(nc),
                            buffer_size_limit=2048,
                        )
                        max_scan_kernel[(M2 // bm3,)](
                            src,
                            part_val,
                            best_c,
                            out_flat,
                            out_idx_flat,
                            M2,
                            N,
                            bm3,
                            CHUNK,
                            buffer_size_limit=2048,
                        )
                    if not keepdim:
                        out_value = torch.squeeze(out_value, dim)
                        out_index = torch.squeeze(out_index, dim)
                    Max_out = namedtuple("max", ["values", "indices"])
                    return Max_out(values=out_value, indices=out_index)

        tile = _pick_fast_tile(M2, N, is_fp32)
        if tile is not None:
            block_m, block_n = tile
            grid_m = M2 // block_m
            need_mask = False
            perm = [d for d in range(inp.dim()) if d != dim] + [dim]
            view = inp.permute(perm)
            if view.is_contiguous():
                src = view
            else:
                src = torch.empty(list(view.shape), dtype=dtype, device=inp.device)
                with torch_device_fn.device(inp.device):
                    if not tle_copy(view, src):
                        torch.ops.aten._copy_from(view, src, False)
            nc = triton.cdiv(N, block_n)
            part_val = torch.empty((M2, nc), dtype=torch.float32, device=inp.device)
            best_c = torch.empty((M2,), dtype=torch.int32, device=inp.device)
            out_flat = out_value.reshape(-1)
            out_idx_flat = out_index.reshape(-1)
            with torch_device_fn.device(inp.device):
                max_split_kernel[(grid_m,)](
                    src,
                    part_val,
                    M2,
                    N,
                    block_m,
                    block_n,
                    need_mask,
                    buffer_size_limit=2048,
                )
                max_chunk_kernel_col[(grid_m,)](
                    part_val,
                    best_c,
                    M2,
                    nc,
                    block_m,
                    triton.next_power_of_2(nc),
                    buffer_size_limit=2048,
                )
                max_scan_kernel_col[(grid_m,)](
                    src,
                    part_val,
                    best_c,
                    out_flat,
                    out_idx_flat,
                    M2,
                    N,
                    block_m,
                    block_n,
                    buffer_size_limit=2048,
                )
            if not keepdim:
                out_value = torch.squeeze(out_value, dim)
                out_index = torch.squeeze(out_index, dim)
            Max_out = namedtuple("max", ["values", "indices"])
            return Max_out(values=out_value, indices=out_index)

    # Integer path (or float shapes with no mask-free fast tile): constexpr
    # masked kernel for small N, two-stage for large N.
    inp = inp.contiguous()

    with torch_device_fn.device(inp.device):
        if N > MAX_TILE_N:
            block_n = STAGE_BLOCK_N
            block_m = STAGE_BLOCK_M
            num_chunks = triton.cdiv(N, block_n)
            part_val = torch.empty(
                (M * num_chunks * K,), dtype=inp.dtype, device=inp.device
            )
            part_idx = torch.empty(
                (M * num_chunks * K,), dtype=torch.int64, device=inp.device
            )
            grid1 = (triton.cdiv(M, block_m), num_chunks, K)
            max_stage1[grid1](
                inp, part_val, part_idx, M, N, K, num_chunks, block_m, block_n
            )
            block_c = triton.next_power_of_2(num_chunks)
            grid2 = (triton.cdiv(M, block_m), K)
            max_stage2[grid2](
                part_val,
                part_idx,
                out_value,
                out_index,
                M,
                K,
                num_chunks,
                block_m,
                block_c,
            )
        else:
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K)  # noqa: E731
            max_kernel[grid](inp, out_value, out_index, M, N, K, isCloseCoreTiling=True)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)
    Max_out = namedtuple("max", ["values", "indices"])
    return Max_out(values=out_value, indices=out_index)
