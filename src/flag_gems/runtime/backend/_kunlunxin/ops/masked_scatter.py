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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# masked_scatter is  out[p] = where(mask[p], source[cumsum(mask)[p]-1], inp[p]).
#
# N > 4096 fast path (this file): a VENDOR-STYLE 2-LAUNCH, 768-thread, contiguous
# PARTITION model (ported from masked_scatter.xpu stage_1/stage_2):
#   stage_1: each of GRID*64 = 768 threads owns a 64-aligned contiguous partition
#            of N, counts its Trues, then an intra-cluster __shared__ prefix writes
#            per-thread counts tcnt[768] and per-cluster totals ctot[12] -- NO big
#            cumsum, NO base[] GM round-trip.
#   stage_2: each thread derives its global source base = sum(ctot[0..cluster-1])
#            + intra-cluster prefix(tcnt), then scatters its partition with the
#            rolling-source compaction loop.
# This keeps all 768 threads busy even for tiny N (fixes the 256^2/1024^2 launch-
# overhead floor) and removes a whole launch + two GM arrays vs the old 3-launch
# (count -> counts[] -> scan -> base[] -> where).  GRID=12 == the 12 physical
# clusters of this XPU (sibling histc tle.raw runs GRID=12 with exact sums, which
# requires all 768 workers; torch multi_processor_count==8 under-reports).


@libentry()
@triton.jit
def masked_scatter_single_block_kernel(
    out_ptr, inp_ptr, mask_ptr, src_ptr, N, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N
    inp_val = tl.load(inp_ptr + offsets, mask=block_mask, other=0)
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    src_indices = tl.cumsum(mask_val.to(tl.int32), axis=0) - 1
    active = block_mask & mask_val
    src_val = tl.load(src_ptr + src_indices, mask=active, other=0)
    out_val = tl.where(mask_val, src_val, inp_val)
    tl.store(out_ptr + offsets, out_val, mask=block_mask)


_MS_GRID = 12  # 12 physical clusters; grid=16/24 would drop clusters -> never exceed
_MS_WT = {"int": 1024, "short": 1024}  # elements staged per DMA chunk
_MS_KT = {"int": 256, "short": 512}  # rolling source-window chunk (elements)
_MS_TLE = {"init": False, "ok": False, "stage1": None, "stage2": {}}


def _ms_stage1_payload():
    """Count Trues per 768-thread contiguous partition + intra-cluster prefix.

    Writes tcnt[768] (per-thread True count; core0 of each cluster writes its 64
    counts as one 256 B LM2GM) and ctot[12*16] (per-cluster total, padded to a
    64 B store).  Both stores are >= 64 B -- a sub-64 B store to a _global_ptr_
    wedges the card.  Every thread (incl. empty partitions, c=0) reaches sync_all.
    """
    return rf"""
#include "xpu/kernel/xtdk.h"
#define WT {_MS_WT["int"]}
#define GRID {_MS_GRID}
extern "C" __device__ void ms_stage1(_global_ptr_ int *tcnt,
    _global_ptr_ int *ctot, _global_ptr_ const unsigned char *mask, int N) {{
  __local__ __attribute__((aligned(64))) unsigned char lmask[WT];
  __shared__ int sm[64];
  int ncore = core_num();
  int cl = cluster_id();
  int co = core_id();
  int nthreads = GRID * ncore;
  int tid = cl * ncore + co;
  long nunit = ((long)N + 63) / 64;
  long upt = nunit / nthreads;
  long rem = nunit - upt * nthreads;
  long ustart = (long)tid * upt + (tid < rem ? tid : rem);
  long usize = upt + (tid < rem ? 1 : 0);
  long start = ustart * 64;
  long end = start + usize * 64;
  if (end > (long)N) end = (long)N;
  int c = 0;
  for (long p = start; p < end; p += WT) {{
    int w = (end - p < WT) ? (int)(end - p) : WT;
    GM2LM(mask + p, lmask, w);
    __local__ unsigned int *w4 = (__local__ unsigned int *)lmask;
    int nw = w >> 2;
    for (int k = 0; k < nw; ++k) {{
      unsigned int v = w4[k];
      c += (v & 0xff) + ((v >> 8) & 0xff) + ((v >> 16) & 0xff) + (v >> 24);
    }}
  }}
  sm[co] = c;
  mfence();
  sync_all();
  if (co == 0) {{
    __local__ int lt[64];
    __local__ int ct[16];
    int acc = 0;
    for (int i = 0; i < ncore; ++i) {{ lt[i] = sm[i]; acc += sm[i]; }}
    for (int i = ncore; i < 64; ++i) lt[i] = 0;
    LM2GM(lt, tcnt + cl * 64, 64 * 4);
    for (int i = 0; i < 16; ++i) ct[i] = 0;
    ct[0] = acc;
    LM2GM(ct, ctot + cl * 16, 16 * 4);
  }}
  mfence();
}}
"""


def _ms_stage2_payload(elt, wt, kt, name):
    """Scatter each 768-thread contiguous partition; single-buffered.

    global src base = sum(ctot[0..cl-1]) + intra-cluster prefix(tcnt).  The
    rolling source window advances across the WHOLE partition (not per block),
    capped by this thread's own True count ctid so src reads never exceed the
    valid [gbase, gbase+ctid) range.  Nothing writes a _global_ptr_ scalar; the
    output store is one contiguous LM2GM per WT chunk (chunks are 64-aligned).
    """
    return rf"""
#include "xpu/kernel/xtdk.h"
#define WT {wt}
#define KT {kt}
#define GRID {_MS_GRID}
extern "C" __device__ void {name}(_global_ptr_ {elt} *out,
    _global_ptr_ const {elt} *inp,
    _global_ptr_ const unsigned char *mask,
    _global_ptr_ const {elt} *src,
    _global_ptr_ const int *tcnt,
    _global_ptr_ const int *ctot,
    int N, int ELT) {{
  __local__ {elt} lout[WT];
  __local__ __attribute__((aligned(64))) unsigned char lmask[WT];
  __local__ {elt} lsrc[KT];
  __local__ int lct[GRID * 16];
  __local__ int ltc[64];
  int ncore = core_num();
  int cl = cluster_id();
  int co = core_id();
  int nthreads = GRID * ncore;
  int tid = cl * ncore + co;
  long nunit = ((long)N + 63) / 64;
  long upt = nunit / nthreads;
  long rem = nunit - upt * nthreads;
  long ustart = (long)tid * upt + (tid < rem ? tid : rem);
  long usize = upt + (tid < rem ? 1 : 0);
  long start = ustart * 64;
  long end = start + usize * 64;
  if (end > (long)N) end = (long)N;
  if (start < end) {{
    GM2LM(ctot, lct, GRID * 16 * 4);
    GM2LM(tcnt + cl * ncore, ltc, ncore * 4);
    int gbase = 0;
    for (int c = 0; c < cl; ++c) gbase += lct[c * 16];
    for (int j = 0; j < co; ++j) gbase += ltc[j];
    int ctid = ltc[co];
    int gcons = 0, bufpos = 0;
    int buflen = (ctid < KT) ? ctid : KT;
    if (buflen > 0) GM2LM(src + gbase, lsrc, buflen * ELT);
    for (long p = start; p < end; p += WT) {{
      int w = (end - p < WT) ? (int)(end - p) : WT;
      GM2LM(inp + p, lout, w * ELT);
      GM2LM(mask + p, lmask, w);
      for (int i = 0; i < w; ++i) {{
        if (lmask[i]) {{
          if (bufpos == buflen) {{
            gcons += buflen;
            int rem2 = ctid - gcons;
            buflen = (rem2 < KT) ? rem2 : KT;
            bufpos = 0;
            if (buflen > 0) GM2LM(src + gbase + gcons, lsrc, buflen * ELT);
          }}
          lout[i] = lsrc[bufpos];
          ++bufpos;
        }}
      }}
      LM2GM(lout, out + p, w * ELT);
    }}
  }}
  mfence();
}}
"""


def _ms_tle_init():
    if _MS_TLE["init"]:
        return _MS_TLE["ok"]
    _MS_TLE["init"] = True
    try:
        import triton as _triton
        import triton.experimental.tle as _tle

        s1ns = {"_tle": _tle}
        exec(
            "@_tle.raw.dialect('xpu3', source=_SRC)\n"
            "def ms_stage1(tcnt, ctot, mask, N):\n"
            "    ...\n",
            {"_tle": _tle, "_SRC": _ms_stage1_payload()},
            s1ns,
        )
        _s1callee = s1ns["ms_stage1"]

        @_triton.jit
        def _s1(T, C, M, N):
            _tle.raw.call(_s1callee, (T, C, M, N))

        _MS_TLE["stage1"] = _s1

        for elt in ("int", "short"):
            sym = f"ms_stage2_{elt}"
            ns = {"_tle": _tle}
            exec(
                "@_tle.raw.dialect('xpu3', source=_SRC)\n"
                f"def {sym}(out, inp, mask, src, tcnt, ctot, N, ELT):\n"
                "    ...\n",
                {
                    "_tle": _tle,
                    "_SRC": _ms_stage2_payload(elt, _MS_WT[elt], _MS_KT[elt], sym),
                },
                ns,
            )
            callee = ns[sym]

            def _mk(callee=callee):
                @_triton.jit
                def _k(O, I, M, S, T, C, N, ELT):
                    _tle.raw.call(callee, (O, I, M, S, T, C, N, ELT))

                return _k

            _MS_TLE["stage2"][elt] = _mk()
        _MS_TLE["ok"] = True
    except Exception as _e:  # pragma: no cover - keep backend importable
        logger.debug("masked_scatter tle.raw init failed: %r", _e)
        _MS_TLE["ok"] = False
    return _MS_TLE["ok"]


@libentry()
@triton.jit
def mask_block_count_kernel(mask_ptr, counts_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    sel = tl.load(mask_ptr + offsets, mask=offsets < N, other=0).to(tl.int32)
    tl.store(counts_ptr + pid, tl.sum(sel, axis=0))


@libentry()
@triton.jit
def masked_scatter_where_kernel(
    out_ptr, inp_ptr, mask_ptr, src_ptr, base_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    block_mask = offsets < N
    base = tl.load(base_ptr + pid)
    mask_val = tl.load(mask_ptr + offsets, mask=block_mask, other=0).to(tl.int1)
    src_indices = base + tl.cumsum(mask_val.to(tl.int32), axis=0) - 1
    active = block_mask & mask_val
    inp_val = tl.load(inp_ptr + offsets, mask=block_mask, other=0)
    src_val = tl.load(src_ptr + src_indices, mask=active, other=0)
    tl.store(out_ptr + offsets, tl.where(mask_val, src_val, inp_val), mask=block_mask)


def _tle_ms_where(out, inp, mask, src, N):
    """2-launch partition masked_scatter; None -> defer to the codegen kernels."""
    try:
        if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return None
        if N >= 2**31:
            return None
        if not (
            out.is_contiguous()
            and inp.is_contiguous()
            and mask.is_contiguous()
            and src.is_contiguous()
        ):
            return None
        if not _ms_tle_init():
            return None
        elt = "int" if inp.dtype == torch.float32 else "short"
        s2 = _MS_TLE["stage2"].get(elt)
        if s2 is None:
            return None
        u8mask = mask.view(torch.uint8)
        nthreads = _MS_GRID * 64
        tcnt = torch.empty(nthreads, dtype=torch.int32, device=inp.device)
        ctot = torch.empty(_MS_GRID * 16, dtype=torch.int32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _MS_TLE["stage1"][(_MS_GRID,)](tcnt, ctot, u8mask, N)
            if elt == "int":
                O, I, S = out, inp, src
            else:
                O = out.view(torch.int16)
                I = inp.view(torch.int16)
                S = src.view(torch.int16)
            s2[(_MS_GRID,)](O, I, u8mask, S, tcnt, ctot, N, inp.element_size())
        return out
    except Exception as _e:  # pragma: no cover - never break correctness
        logger.debug("masked_scatter tle path deferred: %r", _e)
        return None


def masked_scatter_impl(out, inp, mask, source, N):
    flat_out = out.ravel()
    flat_inp = inp.ravel()
    flat_mask = mask.ravel()
    flat_src = source.ravel()
    # an all-False mask yields an empty `source`; guard so a zero-size buffer
    # never reaches the kernels (values are discarded by where for False).
    if flat_src.numel() == 0:
        flat_src = torch.zeros(1, dtype=source.dtype, device=source.device)

    if N == 0:
        return out

    if N <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16
        with torch_device_fn.device(inp.device):
            masked_scatter_single_block_kernel[(1,)](
                flat_out,
                flat_inp,
                flat_mask,
                flat_src,
                N,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        return out

    # 2-launch partition fast path.  Returns None -> defer to the codegen kernels.
    res = _tle_ms_where(flat_out, flat_inp, flat_mask, flat_src, N)
    if res is not None:
        return out

    BLOCK_SIZE = 8192
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    idx_dtype = torch.int32 if N < 2**31 else torch.int64
    with torch_device_fn.device(inp.device):
        counts = torch.empty(n_blocks, dtype=idx_dtype, device=inp.device)
        mask_block_count_kernel[(n_blocks,)](
            flat_mask, counts, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=16
        )
        base = torch.zeros(n_blocks, dtype=idx_dtype, device=inp.device)
        torch.cumsum(counts[:-1], dim=0, out=base[1:])
        masked_scatter_where_kernel[(n_blocks,)](
            flat_out,
            flat_inp,
            flat_mask,
            flat_src,
            base,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=16,
        )
    return out


def masked_scatter(inp, mask, source):
    logger.debug("GEMS_KUNLUNXIN MASKED_SCATTER")

    assert broadcastable(
        inp.shape, mask.shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"

    _, mask = torch.broadcast_tensors(inp, mask)

    out = torch.empty_like(inp)
    if not out.is_contiguous():
        out = out.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    if not source.is_contiguous():
        source = source.contiguous()
    if not inp.is_contiguous():
        inp = inp.contiguous()

    N = out.numel()
    masked_scatter_impl(out, inp, mask, source, N)
    return out


def masked_scatter_(inp, mask, source):
    logger.debug("GEMS_KUNLUNXIN MASKED_SCATTER_")

    assert broadcastable(inp.shape, mask.shape)
    _, mask = torch.broadcast_tensors(inp, mask)

    if not inp.is_contiguous():
        raise RuntimeError(
            "in-place operation currently requires contiguous input tensor. "
        )

    mask = mask if mask.is_contiguous() else mask.contiguous()
    source = source if source.is_contiguous() else source.contiguous()

    N = inp.numel()
    masked_scatter_impl(inp, inp, mask, source, N)
    return inp
