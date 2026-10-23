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
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops.le_ import le_ as _generic_le_
from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

# The flagtree compiler in the CI image rejects TRITONXPU_COMPARE_FUSION on the
# wide native kernels (cmpf(ui32) verifier error); detect it once (same as
# less_equal.py) and skip the fusion flag only on that compiler.
_IS_FLAGTREE = os.path.exists(
    os.path.join(os.path.dirname(triton.__file__), "FLAGTREE_BACKEND")
)


config_ = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def le_func(x, y):
    return x.to(tl.float32) <= y


def le(A, B):
    logger.debug("GEMS_KUNLUNXIN LE")
    # Native fused-compare fast path (mirrors less_equal.py, which is the same
    # x <= y). Contiguous same-shape float tensor-tensor writes bool directly,
    # no fp32 buffer / _copy_from. Generic path otherwise.
    numel = A.numel()
    if (
        A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and A.dtype == B.dtype
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and 0 < numel
    ):
        if numel <= _LE_TENSOR_FAST_MAX:
            return _le_tensor_native(A, B, numel, _le_tensor_adaptive_tile(numel))
        return _le_tensor_large(A, B, numel)
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = le_func(A, B)
    if not _IS_FLAGTREE:
        del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


# ---------------------------------------------------------------------------
# le tensor-tensor fast path: native fused compare (x <= y) writing bool
# directly. Mirrors less_equal.py (identical semantics). Tiles / masked-ceiling
# / launch opts are the same values proven there.
_LE_TENSOR_FAST_TILE = 131072
_LE_TENSOR_FAST_GRID = 8
_LE_TENSOR_FAST_TILE_MIN = 4096
_LE_TENSOR_FAST_MAX = 1 << 18
_LE_TENSOR_MASKED_TILE_MAX = 16384
_LE_BUFFER_SIZE_LIMIT = 8192
_LE_FAST_LAUNCH_OPTS = {
    "buffer_size_limit": _LE_BUFFER_SIZE_LIMIT,
    "unroll_num": 16,
    "isCloseMemoryAsync": False,
}
_LE_MASKED_LAUNCH_OPTS = {
    "unroll_num": 16,
    "isCloseMemoryAsync": False,
}


def _le_tensor_adaptive_tile(numel):
    return max(
        _LE_TENSOR_FAST_TILE_MIN,
        triton.next_power_of_2(
            (numel + _LE_TENSOR_FAST_GRID - 1) // _LE_TENSOR_FAST_GRID
        ),
    )


@triton.jit
def le_tensor_native_kernel(out_ptr, x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    r = tl.load(x_ptr + tid) <= tl.load(y_ptr + tid)
    tl.store(out_ptr + tid, r.to(tl.int8))


@triton.jit
def le_tensor_native_masked_kernel(out_ptr, x_ptr, y_ptr, numel, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    tid_c = tl.minimum(tid, numel - 1)
    r = (tl.load(x_ptr + tid_c) <= tl.load(y_ptr + tid_c)) & mask
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _le_tensor_native(A, B, numel, tile=_LE_TENSOR_FAST_TILE):
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    y = B.reshape(-1)
    masked = numel % tile != 0
    if masked and tile > _LE_TENSOR_MASKED_TILE_MAX:
        tile = _LE_TENSOR_MASKED_TILE_MAX
    grid = (math.ceil(numel / tile),) if masked else (numel // tile,)
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    launch_opts = {} if _IS_FLAGTREE else _LE_FAST_LAUNCH_OPTS
    masked_opts = {} if _IS_FLAGTREE else _LE_MASKED_LAUNCH_OPTS
    try:
        if masked:
            le_tensor_native_masked_kernel[grid](
                out, x, y, numel, TILE=tile, num_warps=4, **masked_opts
            )
        else:
            le_tensor_native_kernel[grid](
                out, x, y, TILE=tile, num_warps=4, **launch_opts
            )
    finally:
        if not _IS_FLAGTREE:
            del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return out


@triton.jit
def le_tensor_native_tail_kernel(
    out_ptr, x_ptr, y_ptr, tail, offset, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = offset + pid * TILE + tl.arange(0, TILE)
    mask = tid < offset + tail
    tid_c = tl.minimum(tid, offset + tail - 1)
    r = (tl.load(x_ptr + tid_c) <= tl.load(y_ptr + tid_c)) & mask
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _le_tensor_large(A, B, numel):
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    y = B.reshape(-1)
    tail = numel % _LE_TENSOR_FAST_TILE
    bulk = numel - tail
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    launch_opts = {} if _IS_FLAGTREE else _LE_FAST_LAUNCH_OPTS
    masked_opts = {} if _IS_FLAGTREE else _LE_MASKED_LAUNCH_OPTS
    try:
        if bulk:
            le_tensor_native_kernel[(bulk // _LE_TENSOR_FAST_TILE,)](
                out, x, y, TILE=_LE_TENSOR_FAST_TILE, num_warps=4, **launch_opts
            )
        if tail:
            le_tensor_native_tail_kernel[
                (math.ceil(tail / _LE_TENSOR_MASKED_TILE_MAX),)
            ](
                out,
                x,
                y,
                tail,
                bulk,
                TILE=_LE_TENSOR_MASKED_TILE_MAX,
                num_warps=4,
                **masked_opts,
            )
    finally:
        if not _IS_FLAGTREE:
            del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return out


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def le_func_scalar(x, y):
    return x.to(tl.float32) <= y


def le_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LE_SCALAR")
    numel = A.numel()
    dtype = A.dtype
    if (
        A.is_contiguous()
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=dtype).item())
    ):
        if numel >= _LE_SCALAR_FAST_TILE and numel % _LE_SCALAR_FAST_TILE == 0:
            return _le_scalar_native(A, float(B), numel, masked=False)
        if numel >= _LE_SCALAR_MASKED_MIN and numel % _LE_SCALAR_FAST_TILE != 0:
            return _le_scalar_native(A, float(B), numel, masked=True)
        # Small / thin shapes (numel < FAST_TILE): route to the native fused
        # kernel with an adaptive small TILE instead of the ~10x-slower generic
        # pointwise path. Tile hugs numel (next_pow2, floor 1024) so we don't
        # launch a 131072-lane program for a few-K-element tensor.
        if 0 < numel < _LE_SCALAR_FAST_TILE:
            tile = min(_LE_SCALAR_FAST_TILE, max(1024, triton.next_power_of_2(numel)))
            return _le_scalar_native(
                A, float(B), numel, masked=(numel % tile != 0), tile=tile
            )
    res = le_func_scalar(A, B)
    return res


_LE_SCALAR_FAST_TILE = 131072
_LE_SCALAR_MASKED_MIN = 1 << 20

# ---------------------------------------------------------------------------
# bf16 note: xpu3 has no native bf16 compare, so bf16 widens to fp32 (16-lane)
# and large shapes stay compiler-bound (~0.56); fp16/fp32 reach the fused path.


@triton.jit
def le_scalar_native_kernel(
    out_ptr, x_ptr, scalar, TILE: tl.constexpr, DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    r = x <= scalar.to(DTYPE)
    tl.store(out_ptr + tid, r.to(tl.int8))


@triton.jit
def le_scalar_native_masked_kernel(
    out_ptr, x_ptr, scalar, numel, TILE: tl.constexpr, DTYPE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    mask = tid < numel
    x = tl.load(x_ptr + tid, mask=mask)
    r = x <= scalar.to(DTYPE)
    tl.store(out_ptr + tid, r.to(tl.int8), mask=mask)


def _le_scalar_native(A, scalar, numel, masked, tile=_LE_SCALAR_FAST_TILE):
    # Single-kernel native compare: writes the bool result directly, so the
    # fp32 intermediate buffer and _copy_from pass of the saturating recipe
    # are gone. Env must be set before the (first) launch so the fusion pass
    # sees it at compile time (same set/del pattern as le above).
    out = torch.empty_like(A, dtype=torch.bool)
    x = A.reshape(-1)
    grid = (math.ceil(numel / tile),) if masked else (numel // tile,)
    DTYPE = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[A.dtype]
    if not _IS_FLAGTREE:
        os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    try:
        if masked:
            le_scalar_native_masked_kernel[grid](
                out,
                x,
                scalar,
                numel,
                TILE=tile,
                DTYPE=DTYPE,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
        else:
            le_scalar_native_kernel[grid](
                out,
                x,
                scalar,
                TILE=tile,
                DTYPE=DTYPE,
                num_warps=4,
                buffer_size_limit=8192,
                unroll_num=16,
                isCloseMemoryAsync=False,
            )
    finally:
        if not _IS_FLAGTREE:
            del os.environ["TRITONXPU_COMPARE_FUSION"]
        del os.environ["TRITONXPU_FP16_FAST"]
    return out


# ---------------------------------------------------------------------------
# le_ (in-place x.le_(y)): saturating fp32 le = 1 - min(1,max(0,(x-y)*1e64))
# written back into x, no i1. In-place-safe config (DEFAULT isCloseMemoryAsync)
# avoids the noc-idle-timeout deadlock async-copy has under aliasing.
config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_inplace_)
@triton.jit
def le_func_tensor_inplace(x, y):
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def le_(A, B):
    logger.debug("GEMS_KUNLUNXIN LE__TENSOR")
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    numel = A.numel()
    if A.is_contiguous() and A.dtype in (torch.float16, torch.float32, torch.bfloat16):
        if (
            A.dtype in (torch.float16, torch.float32)
            and B.is_contiguous()
            and B.dtype == A.dtype
            and A.shape == B.shape
            and numel >= _LE_TENSOR_INPLACE_FAST_TILE * _LE_TENSOR_INPLACE_MIN_GRID
            and numel % _LE_TENSOR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _le_tensor_inplace_fast(A, B, numel)
        le_func_tensor_inplace(A, B, out0=A)
        return A
    # Everything else (non-float dtype, non-contiguous, ...) keeps the
    # original generic in-place path, behavior unchanged.
    return _generic_le_(A, B)


# in-place alias safety: the fast kernel writes into the SAME tensor it
# reads, so it must keep the DEFAULT isCloseMemoryAsync (True = async copy
# closed); passing False with in-place aliasing is the documented "noc idle
# timeout" deadlock, same as gt.py's in-place fast path note.
_LE_TENSOR_INPLACE_FAST_TILE = 131072
_LE_TENSOR_INPLACE_MIN_GRID = 128


@triton.jit
def le_tensor_inplace_fast_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    t = (x.to(tl.float32) - y.to(tl.float32)) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, 1.0 - t)


def _le_tensor_inplace_fast(A, B, numel):
    grid = (numel // _LE_TENSOR_INPLACE_FAST_TILE,)
    le_tensor_inplace_fast_kernel[grid](
        A,
        B,
        TILE=_LE_TENSOR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A


# ---------------------------------------------------------------------------
# le_scalar_ (in-place x.le_(scalar)): saturating fp32 recipe le = 1 - min(1,
# max(0, (x-s)*1e64)) written back into x, no i1. In-place-safe config (DEFAULT
# isCloseMemoryAsync) avoids the noc-idle-timeout deadlock that async-copy has
# under aliasing. Mirrors less_equal_scalar_ (same x <= s semantics).
@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_inplace_,
)
@triton.jit
def le_func_scalar_inplace(x, y):
    t = (x.to(tl.float32) - y) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    return 1.0 - t


def le_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LE_SCALAR_")
    numel = A.numel()
    if (
        A.is_contiguous()
        and A.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and float(B) == float(torch.tensor(float(B), dtype=A.dtype).item())
    ):
        if (
            A.dtype in (torch.float16, torch.float32)
            and numel >= _LE_SCALAR_INPLACE_FAST_TILE * _LE_SCALAR_INPLACE_MIN_GRID
            and numel % _LE_SCALAR_INPLACE_FAST_TILE == 0
        ):
            # exact-multiple flat tiles: no mask at all; grid fixed.
            return _le_scalar_inplace_fast(A, float(B))
        le_func_scalar_inplace(A, B, out0=A)
        return A
    return le_func_scalar(A, B, out0=A)


# in-place alias safety: the fast kernel writes into the SAME tensor it reads,
# so it must keep the DEFAULT isCloseMemoryAsync (True = async copy closed);
# passing False with in-place aliasing is the documented noc idle timeout
# deadlock, same as the le_ / gt_ in-place fast path note.
_LE_SCALAR_INPLACE_FAST_TILE = 131072
_LE_SCALAR_INPLACE_MIN_GRID = 128


@triton.jit
def le_scalar_inplace_fast_kernel(x_ptr, scalar, TILE: tl.constexpr):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    t = (x.to(tl.float32) - scalar) * 1.0e32
    t = t * 1.0e32
    t = tl.maximum(0.0, t)
    t = tl.minimum(1.0, t)
    tl.store(x_ptr + tid, 1.0 - t)


def _le_scalar_inplace_fast(A, scalar):
    grid = (A.numel() // _LE_SCALAR_INPLACE_FAST_TILE,)
    le_scalar_inplace_fast_kernel[grid](
        A,
        scalar,
        TILE=_LE_SCALAR_INPLACE_FAST_TILE,
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=True,
    )
    return A
