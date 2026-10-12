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
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# --- tle.gpu 2D block-tiled fast path (pure-Triton cluster DMA, no C++) --------
# Each program owns one [BM, BN] block, DMA'd to LM via a TensorDescriptor.
# Dispatch per block vs the diagonal: fully-kept -> pure copy (DMA in/out, no
# compute); fully-dropped -> write zeros (write-only DMA); straddling the
# diagonal -> copy + per-element tl.where. ~7/8 of blocks are pure DMA so the
# element-rate `where` wall is dodged on all but the diagonal band.
#
# BM is pinned to 64: the TLE legacy tile planner requires
# elemsPerCore = BM*BN/core_num(64) <= innermost(BN), i.e. BM <= 64, for the
# pure-copy blocks to compile. BN is adaptive (min(256, N)).
try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE_GPU = True
except Exception:  # pragma: no cover - tle unavailable
    tle = None
    TensorDescriptor = None
    _HAS_TLE_GPU = False

_TLE_DT = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
_TLE_BM = 64


# NOTE: kunlunxin's original triu used the fixed-cluster `M//12` heuristic with
# an in-kernel N loop (slow on large shapes + per-shape first-compile spikes),
# and did NOT override triu_ at all (it fell to the generic slow path).
#
# On this XPU/triton, 2D tiled kernels (`offs_m * N + offs_n` indexing) are not
# proven contiguous by OffsetAnalysis and degrade to discrete access (~1-3 GB/s,
# e.g. a plain [1024,1024] tile copy took ~3ms). The winning primitive is the
# 1D-flat kernel: contiguous load/store with row/col recovered via div/mod. Both
# triu (out-of-place) and triu_ (in-place, in_ptr == out_ptr) are built on it.
# triu keeps col >= row + diag; the complement (col < row + diag) is zeroed.
#
# A pure masked scalar store (`tl.store(ptr, 0.0, mask=...)`) is MISCOMPILED here
# (it ignores the per-lane predicate and zeros the whole block), so we always use
# the load + `tl.where` + store form, which is verified correct.
#
# The flat kernel's cost is dominated by the per-element integer div/mod used to
# recover (row, col) from the flat offset (~112 GB/s, ~4x below pure bandwidth).
# For large N we instead launch one program PER ROW (grid = batch*M): `row` is
# recovered once per program via `pid % M` (one mod per program, not per element)
# and each row streams its N columns as contiguous blocks. Isolation on this XPU:
# [4096,4096] 0.72ms -> 0.52ms, [10000,65536] 27.8ms -> 15.2ms. But for small N
# the row grid is launch-bound (e.g. [100,65536,100] N=100 blows up to 391ms), so
# we gate on N and keep the flat kernel for small last dims.


if _HAS_TLE_GPU:

    @triton.jit
    def _triu_blk_kernel(
        a_desc,
        c_desc,
        Mmat,
        N,
        diag,
        NBCOL: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        # One [BM, BN] block per program. `rl = r0 % Mmat` is the within-matrix
        # row of the block top (handles a batched desc viewed as (B*Mmat, N)).
        # NBCOL = cdiv(N, BN): when N is not a multiple of BN (non-pow2 N padded
        # up to a pow2 BN) or BR is not a multiple of BM (single-matrix row tail),
        # the trailing block is partial and TMA auto-trims cols>=N / rows>=M on
        # both load and store, so padded lanes are never touched in HBM. `cmax`
        # is the largest REAL (unpadded) column in the block, used for the
        # fully-dropped test so padded columns don't skew the dispatch.
        pid = tl.program_id(0)
        r0 = (pid // NBCOL) * BM
        c0 = (pid % NBCOL) * BN
        rl = r0 % Mmat
        cmax = c0 + BN - 1
        cmax = tl.where(cmax < N - 1, cmax, N - 1)
        buf = tle.gpu.alloc([BM, BN], dtype=DTYPE, layout=None, scope=tle.gpu.lmem)
        if c0 >= rl + BM - 1 + diag:
            # whole block kept -> pure copy
            tle.gpu.copy(a_desc, buf, [BM, BN], [r0, c0])
            tle.gpu.copy(buf, c_desc, [BM, BN], [r0, c0])
        elif cmax < rl + diag:
            # whole block dropped -> write zeros
            rid = tl.arange(0, BM)[:, None]
            cid = tl.arange(0, BN)[None, :]
            p = tle.gpu.local_ptr(
                buf, (tl.broadcast_to(rid, (BM, BN)), tl.broadcast_to(cid, (BM, BN)))
            )
            tl.store(p, tl.zeros((BM, BN), dtype=DTYPE))
            tle.gpu.copy(buf, c_desc, [BM, BN], [r0, c0])
        else:
            # straddles the diagonal -> copy + per-element where
            tle.gpu.copy(a_desc, buf, [BM, BN], [r0, c0])
            rid = tl.arange(0, BM)[:, None]
            cid = tl.arange(0, BN)[None, :]
            rr = tl.broadcast_to(rid, (BM, BN))
            cc = tl.broadcast_to(cid, (BM, BN))
            p = tle.gpu.local_ptr(buf, (rr, cc))
            x = tl.load(p)
            keep = (c0 + cc) >= (rl + rr) + diag
            tl.store(p, tl.where(keep, x, tl.zeros((BM, BN), dtype=DTYPE)))
            tle.gpu.copy(buf, c_desc, [BM, BN], [r0, c0])

    @triton.jit
    def _triu_blk_pack_kernel(
        a_desc,
        c_desc,
        Mmat,
        N2,
        diag,
        BM: tl.constexpr,
        BN2: tl.constexpr,
    ):
        # 2-byte dtypes (bf16/fp16) are not vectorized by tle.gpu.copy, so bf16
        # DMA runs ~2-3x below byte-bandwidth. triu only MOVES bits / writes
        # zeros (never computes on kept values), so reinterpret the tensor as
        # int32 (pack 2 columns per word, N2 = N // 2) -> the copy/zero blocks
        # move vectorized 4-byte words. Each packed col cp spans orig cols
        # 2*cp (low 16b) and 2*cp+1 (high 16b); the diagonal straddle block is
        # masked purely with bit ops (0xFFFF low, -65536 == 0xFFFF0000 high).
        pid = tl.program_id(0)
        nbcol = N2 // BN2
        r0 = (pid // nbcol) * BM
        c0 = (pid % nbcol) * BN2
        rl = r0 % Mmat
        buf = tle.gpu.alloc([BM, BN2], dtype=tl.int32, layout=None, scope=tle.gpu.lmem)
        if 2 * c0 >= rl + BM - 1 + diag:
            # smallest orig col in block kept -> whole block kept, pure copy
            tle.gpu.copy(a_desc, buf, [BM, BN2], [r0, c0])
            tle.gpu.copy(buf, c_desc, [BM, BN2], [r0, c0])
        elif 2 * (c0 + BN2 - 1) + 1 < rl + diag:
            # largest orig col in block dropped -> whole block zero
            rid = tl.arange(0, BM)[:, None]
            cid = tl.arange(0, BN2)[None, :]
            p = tle.gpu.local_ptr(
                buf, (tl.broadcast_to(rid, (BM, BN2)), tl.broadcast_to(cid, (BM, BN2)))
            )
            tl.store(p, tl.zeros((BM, BN2), dtype=tl.int32))
            tle.gpu.copy(buf, c_desc, [BM, BN2], [r0, c0])
        else:
            tle.gpu.copy(a_desc, buf, [BM, BN2], [r0, c0])
            rid = tl.arange(0, BM)[:, None]
            cid = tl.arange(0, BN2)[None, :]
            rr = tl.broadcast_to(rid, (BM, BN2))
            cc = tl.broadcast_to(cid, (BM, BN2))
            p = tle.gpu.local_ptr(buf, (rr, cc))
            v = tl.load(p)
            orig_lo = 2 * (c0 + cc)
            row = rl + rr
            keep_lo = orig_lo >= row + diag
            keep_hi = (orig_lo + 1) >= row + diag
            lo = tl.where(keep_lo, v & 0xFFFF, 0)
            hi = tl.where(keep_hi, v & -65536, 0)  # -65536 == 0xFFFF0000 (int32)
            tl.store(p, lo | hi)
            tle.gpu.copy(buf, c_desc, [BM, BN2], [r0, c0])


def _tle_block_eligible(A, diagonal):
    # Contiguous, supported dtype. Block path handles any diag. Non-pow2 N and a
    # single matrix whose M is not a multiple of BM are supported by padding the
    # trailing block up to the pow2 tile and letting TMA trim the overrun.
    if not _HAS_TLE_GPU or A.dtype not in _TLE_DT:
        return None
    if os.environ.get("TRIU_NO_BLOCK"):
        return None
    if not A.is_contiguous():
        return None
    Mmat, N = A.shape[-2:]
    # Block path pays off only when the DMA volume dominates the descriptor /
    # launch overhead. Tiny shapes are host-overhead-bound and see no gain, so
    # leave them on the flat/row path. The flat lean kernel is per-element
    # div/mod-bound and loses badly from ~1M elements up (measured 1024x1024:
    # flat ~0.3x vs block ~0.7-0.8x of aten), so the gate sits at 1M.
    if A.numel() < (1 << 20):
        return None
    if N < _TLE_BM:  # narrow last dim -> flat/row path wins
        return None
    BR = A.numel() // N  # total rows across the batch
    B = BR // Mmat  # number of matrices
    # For a BATCHED input the block rows must align to matrix boundaries so that
    # `rl = r0 % Mmat` is constant within a block; require Mmat % BM == 0. A
    # single matrix has no interior boundary, so any M is fine (row tail is
    # TMA-trimmed).
    if B > 1 and Mmat % _TLE_BM != 0:
        return None
    # Column tile (pow2). Wide matrices DMA better with BN=512 (nbcol stays >= 4
    # so most blocks are still pure copy); narrow ones use 256; non-pow2 N is
    # padded up to the next pow2 (TMA trims back to N).
    if N >= 2048:
        BN = 512
    elif N >= 256:
        BN = 256
    else:
        BN = triton.next_power_of_2(N)
    BM = _TLE_BM
    # Wide matrices (N >= 2048, e.g. 4096x4096): a squarer BM128/BN256 tile beats
    # the flatter BM64/BN512 for every dtype. Measured 4096x4096 (do_bench, card3,
    # err=0): fp32 1.01 -> 1.13, fp16 0.74 -> 0.81, bf16(pack) 0.70 -> 0.81. Needs
    # row alignment to BM so `rl = r0 % Mmat` is constant within a block (a single
    # matrix always qualifies; a batched input needs Mmat % 128 == 0, otherwise it
    # keeps the BM64/BN512 fallback which stays correct for Mmat % 64 batches).
    if N >= 2048 and (B == 1 or Mmat % 128 == 0):
        BM = 128
        BN = 256
    # Batched medium-N shapes (e.g. 64x512x512): a taller/squarer tile cuts the
    # program count and feeds the TMA larger contiguous runs. Measured 64x512x512
    # fp32 0.52x (BM64,BN256) -> 0.65x (BM128,BN128); fp16 0.33 -> 0.41, bf16
    # 0.29 -> 0.39. Only when matrix rows align to the bigger tile (Mmat % 128 ==
    # 0) so `rl = r0 % Mmat` is constant within a block, and only for batched
    # inputs (single-matrix shapes already tile well at BM64 and would regress).
    # Restricted to N < 2048 so it does not clobber the wide-matrix tile above.
    if B > 1 and 256 <= N < 2048 and Mmat % 128 == 0:
        BM = 128
        BN = 128
    nbcol = triton.cdiv(N, BN)
    # bf16 int32-packing (2 cols/word, ~2x on 2-byte DMA) is applied ONLY in the
    # fully-aligned case (no column or row padding) where it is verified exact;
    # packing combined with a single-matrix row-tail trim miscompiles, so padded
    # shapes keep the per-dtype kernel.
    use_pack = (
        A.dtype == torch.bfloat16
        and N % 2 == 0
        and N % BN == 0
        and BR % BM == 0
    )
    return BR, Mmat, N, BM, BN, nbcol, use_pack


def _launch_triu_block(A, out, diagonal, plan):
    BR, Mmat, N, BM, BN, nbcol, use_pack = plan
    nrow = triton.cdiv(BR, BM)
    if use_pack:
        # bf16 DMA is not vectorized by tle.gpu.copy (~2-3x below bandwidth); run
        # the int32-packed kernel (triu only moves bits) for a ~2x bf16 speedup.
        N2 = N // 2
        BN2 = BN // 2
        a2 = A.view(BR, N).view(torch.int32)
        o2 = out.view(BR, N).view(torch.int32)
        a = TensorDescriptor.from_tensor(a2, block_shape=[BM, BN2])
        c = TensorDescriptor.from_tensor(o2, block_shape=[BM, BN2])
        grid = (nrow * nbcol,)
        _triu_blk_pack_kernel[grid](a, c, Mmat, N2, diagonal, BM, BN2)
        return
    a2 = A.view(BR, N)
    o2 = out.view(BR, N)
    a = TensorDescriptor.from_tensor(a2, block_shape=[BM, BN])
    c = TensorDescriptor.from_tensor(o2, block_shape=[BM, BN])
    grid = (nrow * nbcol,)
    _triu_blk_kernel[grid](a, c, Mmat, N, diagonal, nbcol, BM, BN, _TLE_DT[A.dtype])


@triton.jit
def _triu_flat_kernel(
    in_ptr,
    out_ptr,
    total,
    diag,
    MN,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Batched single-pass triu: keep where col >= row + diag, else write 0.
    # `% MN` folds the flat index into one matrix, so this works for any batch.
    # NEED_MASK is False only when total % BLOCK_SIZE == 0 (every block full),
    # which lets the load/store drop the always-true mask (slow masked-memory
    # path on this XPU).
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    matrix_offsets = offsets % MN
    rows = matrix_offsets // N
    cols = matrix_offsets - rows * N
    keep = cols >= rows + diag

    if NEED_MASK:
        mask = offsets < total
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y, mask=mask)
    else:
        x = tl.load(in_ptr + offsets)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y)


@triton.jit
def _triu_flat_2d_kernel(
    in_ptr,
    out_ptr,
    total,
    diag,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Single-matrix fast path: no `% MN`. Works out-of-place (in != out),
    # in-place (in == out), and over a contiguous top-row prefix of one matrix
    # (offsets stay within [0, M*N) so row = offset // N is exact).
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    rows = offsets // N
    cols = offsets - rows * N
    keep = cols >= rows + diag

    if NEED_MASK:
        mask = offsets < total
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y, mask=mask)
    else:
        x = tl.load(in_ptr + offsets)
        y = tl.where(keep, x, 0.0)
        tl.store(out_ptr + offsets, y)


def _check_input(A):
    assert A.dim() >= 2, "Input tensor must have at least 2 dimensions"


@triton.jit
def _triu_row_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    diag,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # One program per row (grid = batch*M). `row = pid % M` is a single mod PER
    # PROGRAM, replacing the flat kernel's per-element div/mod. Each row streams
    # its N columns as contiguous BLOCK_N chunks (block DMA), keeping the entries
    # with col >= row + diag and zeroing the rest.
    # NEED_MASK is False only when N % BLOCK_N == 0 (every chunk full), dropping
    # the always-true mask from the load/store.
    pid = tl.program_id(0)
    row = pid % M
    base = pid * N
    for c0 in range(0, N, BLOCK_N):
        cols = c0 + tl.arange(0, BLOCK_N)
        keep = cols >= row + diag
        if NEED_MASK:
            m = cols < N
            x = tl.load(in_ptr + base + cols, mask=m, other=0.0)
            tl.store(out_ptr + base + cols, tl.where(keep, x, 0.0), mask=m)
        else:
            x = tl.load(in_ptr + base + cols)
            tl.store(out_ptr + base + cols, tl.where(keep, x, 0.0))


# Large flat tiles hide the div/where compute better than small ones on this XPU
# (measured: BLOCK 1024 -> 8192 roughly halves latency on [4096,4096]).
_BLOCK_SIZE = 8192

# Route to the per-row kernel when the last dim is at least this wide. Below it
# the flat kernel wins (row grid becomes launch-bound); at/above it the per-row
# kernel wins by dropping the per-element div/mod. [1024,1024] flat wins,
# [4096,4096] row wins, so the crossover sits between them.
_ROW_N_THRESHOLD = 2048

# Lean fast-path gate: small single-matrix contiguous shapes are host-overhead-
# bound (device exec << python wrapper). For them we skip _check_input, the
# .contiguous() call, _tle_block_eligible probe, band-trick logic AND the
# torch_device_fn.device() context manager, issuing one direct flat-2d launch.
_LEAN_MAX_NUMEL = 1 << 20  # <= 1M elements


def _lean_eligible(A):
    return (
        not os.environ.get("TRIU_NO_LEAN")
        and A.dim() >= 2
        and A.is_contiguous()
        and A.shape[-1] < _ROW_N_THRESHOLD
        and A.numel() <= _LEAN_MAX_NUMEL
        and A.dtype in _TLE_DT
    )


def _lean_flat(input_c, out, diagonal):
    # Minimal direct launch: no device() ctx, no branching. Caller guarantees
    # contiguous, N < _ROW_N_THRESHOLD, numel small. Picks the single-matrix
    # flat-2d kernel (no `% MN`) when the whole tensor is one matrix, else the
    # batched flat kernel. This avoids the slow gems copy_/zero_ and the heavy
    # block descriptor build on tiny host-overhead-bound shapes.
    M, N = input_c.shape[-2:]
    MN = M * N
    total = input_c.numel()
    grid = (triton.cdiv(total, _BLOCK_SIZE),)
    need_mask = total % _BLOCK_SIZE != 0
    if total <= MN:
        _triu_flat_2d_kernel[grid](
            input_c, out, total, diagonal, N, _BLOCK_SIZE, need_mask, num_warps=8
        )
    else:
        _triu_flat_kernel[grid](
            input_c, out, total, diagonal, MN, N, _BLOCK_SIZE, need_mask, num_warps=8
        )


def _launch_flat(input_c, out, diagonal, total=None):
    M, N = input_c.shape[-2:]
    MN = M * N
    if total is None:
        total = input_c.numel()
    if N >= _ROW_N_THRESHOLD:
        # Large N: one program per row (avoids per-element div/mod). `total` may
        # be a top-row prefix (band_hi * N) -> that many rows; row = pid % M.
        num_rows = total // N
        block_n = min(triton.next_power_of_2(N), _BLOCK_SIZE)
        need_mask = N % block_n != 0
        with torch_device_fn.device(input_c.device):
            _triu_row_kernel[(num_rows,)](
                input_c, out, M, N, diagonal, block_n, need_mask, num_warps=8
            )
        return
    grid = (triton.cdiv(total, _BLOCK_SIZE),)
    need_mask = total % _BLOCK_SIZE != 0
    with torch_device_fn.device(input_c.device):
        if total <= MN:
            # single matrix (or a top-row prefix of one) -> skip the `% MN`.
            _triu_flat_2d_kernel[grid](
                input_c,
                out,
                total,
                diagonal,
                N,
                _BLOCK_SIZE,
                need_mask,
                num_warps=8,
            )
        else:
            _triu_flat_kernel[grid](
                input_c,
                out,
                total,
                diagonal,
                MN,
                N,
                _BLOCK_SIZE,
                need_mask,
                num_warps=8,
            )


def _band_bounds(M, N, diagonal):
    # Number of top rows that still contain kept elements for the given
    # diagonal. Rows [band_hi, M) are entirely below the diagonal (all zeros).
    return min(M, max(0, N - diagonal))


def _launch_triu_inplace_contiguous(A, diagonal):
    M, N = A.shape[-2:]
    if diagonal >= N:  # zero everything
        A.zero_()
        return
    # Rows [band_hi, M) are entirely below the diagonal. When that slice is
    # contiguous (2D / batch == 1) we memset it in bulk (vendor bandwidth) and
    # run the flat kernel only over the contiguous band prefix [0, band_hi*N).
    # For batched inputs the slice is strided (zero_()/fill_() would miszero a
    # contiguous block on this XPU), so we fall back to one flat pass over all.
    band_hi = _band_bounds(M, N, diagonal)
    if band_hi < M:
        below = A[..., band_hi:, :]
        if below.is_contiguous():
            below.zero_()
            if band_hi > 0:
                _launch_flat(A, A, diagonal, total=band_hi * N)
            return
    _launch_flat(A, A, diagonal)


def triu(A, diagonal=0):
    logger.debug("GEMS_KUNLUNXIN TRIU")
    # Device-bound, block-tileable shapes (contiguous, wide N, numel >= 1M such
    # as 1024x1024) run far faster on the tle.gpu block path than on the flat
    # lean kernel (block avoids the per-element div/mod). Probe block FIRST so
    # these shapes are not captured by the lean gate below; truly tiny
    # host-overhead-bound shapes fail this probe (numel below the block gate or
    # narrow N) and fall through to the lean path unchanged.
    if A.dim() >= 2:
        Mb, Nb = A.shape[-2:]
        if 1 - Mb < diagonal < Nb:
            plan = _tle_block_eligible(A, diagonal)
            if plan is not None:
                out = torch.empty_like(A)
                _launch_triu_block(A, out, diagonal, plan)
                return out
    # Lean small-shape path: contiguous, narrow N, few elements (single matrix
    # OR batched). Host-overhead-bound here, so bypass all the heavy wrapper
    # machinery (and the slow gems copy_/zero_ for trivial diagonals).
    if _lean_eligible(A):
        out = torch.empty_like(A)
        _lean_flat(A, out, diagonal)
        return out
    _check_input(A)
    M, N = A.shape[-2:]
    # Lean tle.gpu block-tiled fast path: avoids the band-trick / device() ctx
    # overhead. Only for the trivial-exclusive, block-tileable, contiguous case.
    # (Redundant with the top probe for contiguous inputs; kept so the dispatch
    # order below is unchanged for any shape the top probe did not capture.)
    if 1 - M < diagonal < N:
        plan = _tle_block_eligible(A, diagonal)
        if plan is not None:
            out = torch.empty_like(A)
            _launch_triu_block(A, out, diagonal, plan)
            return out
    A = A.contiguous()
    out = torch.empty_like(A)
    if diagonal <= 1 - M:  # keep everything
        out.copy_(A)
        return out
    if diagonal >= N:  # zero everything
        out.zero_()
        return out
    # Symmetric band trick to triu_: rows [band_hi, M) are entirely below the
    # diagonal (all-zero output). When that output slice is contiguous
    # (2D / batch == 1) bulk-memset it and run the flat kernel only over the
    # top band prefix [0, band_hi*N) instead of the whole matrix. Only worth it
    # for large matrices: on tiny inputs the extra zero_() launch outweighs the
    # saved band work.
    band_hi = _band_bounds(M, N, diagonal)
    if band_hi < M and A.numel() >= (1 << 20):
        below = out[..., band_hi:, :]
        if below.is_contiguous():
            below.zero_()
            if band_hi > 0:
                _launch_flat(A, out, diagonal, total=band_hi * N)
            return out
    _launch_flat(A, out, diagonal)
    return out


def triu_(A, diagonal=0):
    logger.debug("GEMS_KUNLUNXIN TRIU_")
    # Device-bound block-tileable shapes: probe the in-place block path first so
    # they are not captured by the lean gate (same rationale as triu()). Each
    # block is read to LM before its region is written back, so in==out is safe.
    if A.dim() >= 2:
        Mb, Nb = A.shape[-2:]
        if 1 - Mb < diagonal < Nb:
            plan = _tle_block_eligible(A, diagonal)
            if plan is not None:
                _launch_triu_block(A, A, diagonal, plan)
                return A
    # Lean small-shape in-place path (in == out): same gate as triu(). Keep the
    # free keep-everything early return (diag <= 1-M) ahead of the kernel so a
    # trivial in-place triu_ does zero work instead of a redundant launch.
    if _lean_eligible(A):
        if diagonal <= 1 - A.shape[-2]:  # keep everything, nothing to zero
            return A
        _lean_flat(A, A, diagonal)
        return A
    _check_input(A)
    M, N = A.shape[-2:]
    if diagonal <= 1 - M:  # keep everything, nothing to zero
        return A
    # Lean tle.gpu block path, in-place (in == out): each block is read to LM
    # before its region is written back, so overlapping in/out is safe.
    if diagonal < N:
        plan = _tle_block_eligible(A, diagonal)
        if plan is not None:
            _launch_triu_block(A, A, diagonal, plan)
            return A
    if A.is_contiguous():
        _launch_triu_inplace_contiguous(A, diagonal)
        return A
    # Non-contiguous: operate on a contiguous copy, then write back with copy_
    # to preserve the original data_ptr and stride (true in-place semantics).
    A_c = A.contiguous()
    _launch_triu_inplace_contiguous(A_c, diagonal)
    A.copy_(A_c)
    return A
