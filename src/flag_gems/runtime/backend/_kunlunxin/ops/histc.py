import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# ---- tle.raw LM local-histogram O(n) fast path --------------------------------
# Each (cluster,core) GM2LM-strides a chunk of the flattened input into LM and
# does single-core scalar `lh[bin]++` (no vector atomic conflict, no local_ptr
# index drop), then writes its per-core BINS partial once; host reduces partials.
# Verified correct UNDER flag_gems.use_gems() (unlike the scatter per-row long
# write, histc writes each partial once at the end -> no use_gems tail corruption).
# Built lazily to keep backend import side-effect free. Non-adaptive shapes/dtypes
# defer to the codegen kernels below (dispatch, not fallback).
_HISTC_GRID = 12  # this XPU has exactly 12 clusters: grid=16/24 silently drop whole
#                   clusters (partial coverage -> wrong sum), grid=12 covers every
#                   element.  12 clusters x 64 cores = 768 cores vs 512 at grid=8
#                   => 1.32x on the per-core scan chain (device-measured).
_HISTC_CORE_MAX = 64  # cores per cluster on this XPU (probed via scatter: GRID*64 covers all)
_HISTC_TLE = {"init": False, "ok": False, "k": None}


@triton.jit
def _histc_combine(partial_ptr, out_ptr, bins, TC: tl.constexpr):
    # Cross-core partial reduction: program b sums partial[:, b] over TC rows.
    # Replaces partial.sum(0) (flag_gems reduce ~0.11ms) with one cheap launch.
    b = tl.program_id(0)
    offs = tl.arange(0, TC)
    vals = tl.load(partial_ptr + offs * bins + b)
    tl.store(out_ptr + b, tl.sum(vals, axis=0))


def _histc_tle_payload():
    return r"""
#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_simd.h"
#define TILE 512
#define BINS_MAX 128
#define GRID 12
extern "C" __device__ void histc_lm_raw(_global_ptr_ float *partial,
    _global_ptr_ const float *inp, int N, int bins, float minv, float maxv) {
  __simd__ float lv0[TILE];          // double-buffered input: async-prefetch the
  __simd__ float lv1[TILE];          // next chunk while binning the current one
  __local__ int   lh[BINS_MAX + 1];  // +1: trash slot (index `bins`) for OOB lanes
  __local__ float lf[BINS_MAX];
  float *bufs[2] = {lv0, lv1};
  for (int b = 0; b <= bins; ++b) lh[b] = 0;
  int ncore = core_num();
  int gcore = cluster_id() * ncore + core_id();
  int tc = GRID * ncore;
  // Per-core chunk sized to spread N over *all* tc cores.  A fixed TILE block
  // leaves only ceil(N/TILE) cores busy for small N, so each busy core runs a
  // full TILE-sized serial chain (~121 cyc/elem) and small N costs as much as
  // a TILE-sized N.  chunk is rounded to 16 floats (64 B) for DMA granularity
  // and capped at TILE so large N degenerates to the old block-strided walk.
  int chunk = (N + tc - 1) / tc;
  if (chunk < 16) chunk = 16;
  chunk = (chunk + 15) & ~15;
  if (chunk > TILE) chunk = TILE;
  long nchunks = ((long)N + chunk - 1) / chunk;
  float scale = (float)bins / (maxv - minv);
  int bmax = bins - 1;
  int cur = 0;
  long c = gcore;
  if (c < nchunks) {  // prime: async-load this core's first chunk
    long s0 = c * chunk;
    int l0 = (N - s0 < chunk) ? (int)(N - s0) : chunk;
    GM2LM_ASYNC(inp + s0, bufs[cur], l0 * sizeof(float));
  }
  for (; c < nchunks; c += tc) {
    long start = c * chunk;
    int len = (N - start < chunk) ? (int)(N - start) : chunk;
    mfence_lm_gm();  // bufs[cur] load now complete
    long cn = c + tc;  // async-prefetch next chunk, overlapping the compute below
    if (cn < nchunks) {
      long sn = cn * chunk;
      int ln = (N - sn < chunk) ? (int)(N - sn) : chunk;
      GM2LM_ASYNC(inp + sn, bufs[1 - cur], ln * sizeof(float));
    }
    float *buf = bufs[cur];
    int *lvi = (int *)buf;
    // SIMD bin-ID compute (16 lanes/step); only the final lh[bin]++ stays scalar.
    // The (v-minv)*scale arithmetic, the float->int convert (vfloat2fix_rd ==
    // floor == scalar (int) over the in-range f>=0 domain), the [0,bins-1] clamp
    // and the out-of-range / NaN / v==maxv handling are all vectorised.  OOB lanes
    // (incl NaN/inf) are redirected to a trash slot (index `bins`), so the scalar
    // increment loop stays branchless (`lh[bin]++`, trash discarded).  Masks come
    // straight out of the compare intrinsics as packed ints and feed the `_mh`
    // merges directly -- no lane extract / mask->vector sign-extend (both of which
    // fail to lower on this backend).
    int j = 0;
    for (; j + 16 <= len; j += 16) {
      float32x16_t v = vload_lm_float32x16(buf + j);
      float32x16_t f = svmul_float32x16(scale, svadd_float32x16(-minv, v));
      int32x16_t bi = vfloat2fix_rd(f);                 // floor -> int32
      bi = svmax_int32x16(0, bi);                       // clamp low
      bi = svmin_int32x16(bmax, bi);                    // clamp high -> [0,bins-1]
      int ge_min = svle_float32x16(minv, v);            // v >= minv
      int above = svlt_float32x16(maxv, v);             // v >  maxv
      int oob = ((~ge_min) | above) & 0xffff;           // out of range (NaN/inf too)
      int32x16_t bz = vvxor_int32x16_mh(bi, bi, bi, oob);  // oob ? 0 : bi
      bi = svadd_int32x16_mh(bins, bz, bi, oob);           // oob ? bins : bi
      vstore_lm_int32x16(lvi + j, bi);
    }
    mfence();  // SIMD LM writes above must be visible to the scalar reads below
    for (; j < len; ++j) {  // scalar tail (len % 16), identical semantics
      float v = buf[j];
      int bin = (int)((v - minv) * scale);
      bin = (bin < 0) ? 0 : bin;
      bin = (bin > bmax) ? bmax : bin;
      int in_range = (int)((v >= minv) & (v <= maxv));
      lvi[j] = in_range ? bin : bins;
    }
    for (j = 0; j < len; ++j) lh[lvi[j]] += 1;  // branchless scalar increment
    cur = 1 - cur;
  }
  // LM2GM DMA write of the per-core histogram: ~18x faster than a per-core
  // scalar store loop (44us -> 2.4us for 512 cores x 100 bins, device-measured).
  for (int b = 0; b < bins; ++b) lf[b] = (float)lh[b];
  LM2GM(lf, partial + (long)gcore * bins, bins * sizeof(float));
  mfence();
}
"""


def _histc_tle_init():
    if _HISTC_TLE["init"]:
        return _HISTC_TLE["ok"]
    _HISTC_TLE["init"] = True
    try:
        import triton as _triton
        import triton.experimental.tle as _tle

        @_tle.raw.dialect("xpu3", source=_histc_tle_payload())
        def histc_lm_raw(partial, inp, N, bins, minv, maxv):
            ...

        @_triton.jit
        def _k(P, I, N, B, MN, MX):
            _tle.raw.call(histc_lm_raw, (P, I, N, B, MN, MX))

        _HISTC_TLE["k"] = _k
        _HISTC_TLE["ok"] = True
    except Exception as _e:  # pragma: no cover - keep backend importable
        logger.debug("histc tle.raw init failed: %r", _e)
        _HISTC_TLE["ok"] = False
    return _HISTC_TLE["ok"]


def _histc_tle_lm(inp, bins, min_val, max_val):
    """O(n) LM local-histogram via tle.raw; None -> defer to codegen kernels."""
    try:
        if inp.dtype != torch.float32:
            return None
        if not (0 < bins <= 128):
            return None
        if not (max_val > min_val):
            return None
        n = inp.numel()
        if n <= 0 or n >= 2 ** 31:
            return None
        if not _histc_tle_init():
            return None
        flat = inp.contiguous().view(-1)
        tc_alloc = _HISTC_GRID * _HISTC_CORE_MAX
        partial = torch.empty(tc_alloc, bins, dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _HISTC_TLE["k"][(_HISTC_GRID,)](
                partial, flat, n, bins, float(min_val), float(max_val)
            )
        out = torch.empty(bins, dtype=torch.float32, device=inp.device)
        with torch_device_fn.device(inp.device):
            _histc_combine[(bins,)](partial, out, bins, tc_alloc)
        if out.dtype != inp.dtype:
            out = out.to(inp.dtype)
        return out
    except Exception as _e:  # pragma: no cover - never break correctness
        logger.debug("histc tle path deferred: %r", _e)
        return None



@libentry()
@triton.jit
def histc_bin_main_kernel(
    inp_ptr,
    out_ptr,
    n_main,
    bins,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    b = ext.program_id(0)
    inv_scale = bins / (max_val - min_val)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for start in range(0, n_main, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        v = tl.load(inp_ptr + offs).to(tl.float32)
        idx = tl.floor((v - min_val) * inv_scale).to(tl.int32)
        idx = tl.where(v == max_val, bins - 1, idx)
        in_range = ((v >= min_val) & (v <= max_val)).to(tl.int32)
        hit = (idx == b).to(tl.int32) * in_range
        acc += hit
    total = tl.sum(acc)
    tl.store(out_ptr + b, total.to(tl.float32))


@libentry()
@triton.jit
def histc_bin_tail_kernel(
    inp_ptr,
    out_ptr,
    n_tail,
    off,
    bins,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    b = ext.program_id(0)
    inv_scale = bins / (max_val - min_val)
    offs = off + tl.arange(0, BLOCK_SIZE)
    last = off + n_tail - 1
    cclamp = tl.minimum(offs, last)
    ok = (offs <= last).to(tl.int32)
    v = tl.load(inp_ptr + cclamp).to(tl.float32)
    idx = tl.floor((v - min_val) * inv_scale).to(tl.int32)
    idx = tl.where(v == max_val, bins - 1, idx)
    in_range = ((v >= min_val) & (v <= max_val)).to(tl.int32)
    hit = (idx == b).to(tl.int32) * in_range * ok
    total = tl.sum(hit)
    prev = tl.load(out_ptr + b)
    tl.store(out_ptr + b, total.to(tl.float32) + prev)


@libentry()
@triton.jit
def histc_range_kernel(
    inp_ptr,
    mn_ptr,
    mx_ptr,
    n,
    TILE: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = ext.program_id(0)
    last = n - 1
    mmin = tl.full((TILE,), float("inf"), dtype=tl.float32)
    mmax = tl.full((TILE,), float("-inf"), dtype=tl.float32)
    for start in range(pid * TILE, n, GRID * TILE):
        cols = start + tl.arange(0, TILE)
        cclamp = tl.minimum(cols, last)
        v = tl.load(inp_ptr + cclamp).to(tl.float32)
        mmin = tl.minimum(mmin, v)
        mmax = tl.maximum(mmax, v)
    tl.store(mn_ptr + pid, tl.min(mmin, axis=0))
    tl.store(mx_ptr + pid, tl.max(mmax, axis=0))


@libentry()
@triton.jit
def kern_range_combine(
    mn_ptr,
    mx_ptr,
    out_mn,
    out_mx,
    G: tl.constexpr,
    NG: tl.constexpr,
):
    idx = tl.arange(0, G)
    cclamp = tl.minimum(idx, NG - 1)
    mn = tl.min(tl.load(mn_ptr + cclamp), axis=0)
    mx = tl.max(tl.load(mx_ptr + cclamp), axis=0)
    tl.store(out_mn, mn)
    tl.store(out_mx, mx)


def _data_range(inp):
    n = inp.numel()
    grid = 1
    tiles = triton.cdiv(n, 8192)
    want = triton.cdiv(tiles, 8)
    while grid < 256 and grid * 2 <= want:
        grid *= 2
    mn_part = torch.empty(grid, dtype=torch.float32, device=inp.device)
    mx_part = torch.empty(grid, dtype=torch.float32, device=inp.device)
    out_mn = torch.empty((), dtype=torch.float32, device=inp.device)
    out_mx = torch.empty((), dtype=torch.float32, device=inp.device)
    with torch_device_fn.device(inp.device):
        histc_range_kernel[(grid,)](
            inp,
            mn_part,
            mx_part,
            n,
            TILE=8192,
            GRID=grid,
        )
        g_pow = 1
        while g_pow < grid:
            g_pow *= 2
        kern_range_combine[(1,)](
            mn_part,
            mx_part,
            out_mn,
            out_mx,
            G=g_pow,
            NG=grid,
        )
    return float(out_mn.item()), float(out_mx.item())


def histc(inp, bins=100, min=0, max=0):
    logger.debug("GEMS_KUNLUNXIN HISTC")

    inp = inp.contiguous()

    min_val = float(min)
    max_val = float(max)

    if min_val == 0 and max_val == 0:
        min_val, max_val = _data_range(inp)

    if min_val == max_val:
        out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)
        count = ((inp == min_val) & ~torch.isnan(inp)).sum().item()
        ones = torch.full((1,), count, dtype=inp.dtype, device=inp.device)
        torch.ops.aten._copy_from(ones, out[bins // 2 : bins // 2 + 1], False)
        return out

    out = torch.zeros(bins, dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    for _bs in (8192, 4096, 2048, 1024):
        if n_elements % _bs == 0:
            BLOCK_SIZE = _bs
            break
    grid = (bins,)

    n_main = (n_elements // BLOCK_SIZE) * BLOCK_SIZE
    n_tail = n_elements - n_main

    _tle_out = _histc_tle_lm(inp, bins, min_val, max_val)
    if _tle_out is not None:
        return _tle_out

    with torch_device_fn.device(inp.device):
        if n_main:
            histc_bin_main_kernel[grid](
                inp,
                out,
                n_main,
                bins,
                min_val,
                max_val,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        if n_tail:
            histc_bin_tail_kernel[grid](
                inp,
                out,
                n_tail,
                n_main,
                bins,
                min_val,
                max_val,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out
