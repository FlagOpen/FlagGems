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
from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.block_size_utils import get_block_size_1d
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def kernel_1(inp, target, mid, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr):
    # Full lane-blocks only, grid = M // BLOCK_SIZE: every lane is in-bounds,
    # so the loads are plain and unconditional. Keep it that way: on this
    # backend a masked load does not reliably keep out-of-range lanes out of
    # the reduction (verified with NaN/garbage padding behind the tensor),
    # and a `tl.minimum`-clamped load address miscompiles into an actual
    # out-of-bounds access (device fault). The remainder is reduced by
    # kernel_1_tail below, without masks either.
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_val = tl.load(inp + offset).to(tl.float32)
    target_val = tl.load(target + offset).to(tl.float32)
    sub = inp_val - target_val
    pow_val = sub * sub
    # Reduction.MEAN.value: 1 Reduction.SUM.value: 2
    if reduction == 1:
        sum_val = tl.sum(pow_val) / M
    else:
        sum_val = tl.sum(pow_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def kernel_1_tail(
    inp,
    target,
    mid,
    M,
    base,
    n_keep,
    g,
    BLOCK_SIZE: tl.constexpr,
    reduction: tl.constexpr,
):
    # Reduces the window [M - BLOCK_SIZE, M) -- the last full lane-block,
    # entirely in-bounds. Its trailing `n_keep` lanes are the elements past
    # the full blocks; the leading (BLOCK_SIZE - n_keep) lanes were already
    # counted by kernel_1 and are zeroed BY VALUE (tl.where selects values,
    # never addresses). This is exact for any tensor and never reads out of
    # range.
    lane = tl.arange(0, BLOCK_SIZE)
    inp_val = tl.load(inp + base + lane).to(tl.float32)
    target_val = tl.load(target + base + lane).to(tl.float32)
    sub = inp_val - target_val
    pow_val = sub * sub
    pow_val = tl.where(lane >= BLOCK_SIZE - n_keep, pow_val, 0.0)
    # Reduction.MEAN.value: 1 Reduction.SUM.value: 2
    if reduction == 1:
        sum_val = tl.sum(pow_val) / M
    else:
        sum_val = tl.sum(pow_val)
    tl.store(mid + g, sum_val)


@libentry()
@triton.jit
def kernel_1_unmasked_v2(
    inp, target, mid, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr
):
    # Unmasked stage-1 program over a FULL 32768-lane block
    # (M % BLOCK_SIZE == 0 guaranteed by the host). Masked stage-1 tiles at
    # BLOCK=32768 run at a fraction of the unmasked bandwidth on XPU (masked
    # memory path; e.g. [10000, 65536] mean 14.9ms -> 4.1ms), so fully
    # divisible tensors get the block-DMA path. tl.sum at 32768 lanes is
    # only complete with buffer_size_limit=2048 (enforced at launch); the
    # tile is the largest exact tl.sum tile on this backend.
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_val = tl.load(inp + offset).to(tl.float32)
    target_val = tl.load(target + offset).to(tl.float32)
    sub = inp_val - target_val
    pow_val = sub * sub
    if reduction == 1:
        sum_val = tl.sum(pow_val) / M
    else:
        sum_val = tl.sum(pow_val)
    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def kernel_1_unmasked_pipe(
    inp,
    target,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    CHUNK: tl.constexpr,
    reduction: tl.constexpr,
):
    # Pipelined unmasked stage-1 for the fully divisible big tensors. The
    # single-shot load+reduce at BLOCK >= 131072 leaves the cluster on its
    # reduce tail (two cluster barriers plus a serial 64-entry shared-memory
    # partial sum, measured ~5us per scheduling wave), so the tile is split
    # into CHUNK-lane stages accumulated in registers with num_stages=2
    # software pipelining; the loads of stage i+1 overlap the reduction of
    # stage i (measured 115.8us -> 67.6us on [4096,4096] f16). Only the
    # CHUNK-lane sum is a tl.sum, the partial partials are register adds, so
    # the cross-core tree runs once per program exactly as before. The final
    # sum is exact in fp32 (verified against the CPU reference on
    # [4096,4096]..[1024,65536] for f16/f32/bf16; the chunked order actually
    # matches an f64 reference to within the fp32 sum error, see
    # harness/solution/mse_loss/README.md).
    pid = ext.program_id(0)
    acc = tl.zeros((CHUNK,), dtype=tl.float32)
    for i in tl.range(0, BLOCK_SIZE // CHUNK, num_stages=2):
        offset = pid * BLOCK_SIZE + i * CHUNK + tl.arange(0, CHUNK)
        inp_val = tl.load(inp + offset).to(tl.float32)
        target_val = tl.load(target + offset).to(tl.float32)
        sub = inp_val - target_val
        acc += sub * sub
    sum_val = tl.sum(acc)
    # Reduction.MEAN.value: 1 Reduction.SUM.value: 2
    if reduction == 1:
        sum_val = sum_val / M
    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    # `mid` has BLOCK_MID slots but only the first `mid_size` hold partials;
    # the padding lanes are dropped BY VALUE before the sum -- a value
    # predicate, never an address mask on the load (see the note in kernel_1).
    offset = tl.arange(0, BLOCK_MID)
    mid_val = tl.load(mid + offset).to(tl.float32)
    mid_val = tl.where(offset < mid_size, mid_val, 0.0)
    tl.store(out, tl.sum(mid_val))


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def func(x, y):
    return (x - y) * (x - y)


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


# Unmasked stage-1 tile: the largest tl.sum tile that is exact on this XPU
# with buffer_size_limit=2048 (verified against the CPU reference on
# [4096,4096]..[1024,65536] for f16/f32/bf16; 32768/65536/131072/262144-lane
# single-tile sums are all exact). Measured on the 2-load + reduce pattern of
# this operator the plain 131072-lane tile runs at ~600-1000 GB/s vs the
# pipelined tile (kernel_1_unmasked_pipe) at ~900-1300 GB/s on the big shapes
# (see harness/solution/mse_loss/README.md), so fully divisible tensors get
# the pipelined kernel.
_FULL_BLOCK = 131072
# Pipelined stage-1 tile for the biggest fully divisible tensors: 8 CHUNK
# stages, fewer programs, and the reduce tail amortized over more work
# (measured 73.4us -> 67.6us on [4096,4096] f16 and 378 -> 341us on
# [1024,65536] f32 vs the 131072-lane pipelined tile).
_PIPE_LARGE_BLOCK = 262144
# Tensors at least this many elements take _PIPE_LARGE_BLOCK (16 programs
# minimum); the 4-8 program range [1024,1024]-ish shapes is launch-bound and
# prefers the smaller tile (14.3us vs 10.5us on 1M f16).
_PIPE_LARGE_THRESHOLD = 4194304
# Pipelined stage chunk: 1/8 of _PIPE_LARGE_BLOCK, 1/4 of _FULL_BLOCK. Only
# the CHUNK-lane tl.sum runs the cluster tree; the pipeline overlap wins over
# the single-shot tile for CHUNK < BLOCK (see kernel_1_unmasked_pipe).
_PIPE_CHUNK = 32768
# Secondary exact tile: 2 programs of 32768 lanes beat 1 program of 65536
# lanes on the 65536-element shapes (cluster scheduling), so tensors that are
# only 32768-aligned keep the legacy-size tile.
_V2_BLOCK = 32768
# Legacy full-block size for the non-divisible big shapes: 65536 was the
# fastest of {8192..131072} for the remainder pattern (e.g. [200,40999,3]).
_LEGACY_BLOCK = 65536
# Stage-2 must stay inside the 32768-lane tl.sum ceiling; grow stage-1 blocks
# if the grid would exceed it (legacy MAX_MID rule).
_MAX_MID = 32768
# buffer_size_limit for every kernel launch: the largest exact tl.sum tile
# ceiling on this backend (see the block-size notes above).
_BUFFER_SIZE_LIMIT = 2048


def mse_loss(inp, target, reduction=Reduction.MEAN.value):
    logger.debug("GEMS_KUNLUNXIN MSE_LOSS")
    if reduction == Reduction.NONE.value:
        return func(inp, target)

    inp = inp.contiguous()
    target = target.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    block_size = get_block_size_1d(M, inp.element_size() * 2)

    if (M >= _PIPE_LARGE_THRESHOLD) and (M % _PIPE_LARGE_BLOCK == 0):
        # Largest pipelined tile first: 16+ programs of 262144 lanes. Same
        # exactness argument as _FULL_BLOCK but with the pipelined stages.
        mid_size = M // _PIPE_LARGE_BLOCK
        if mid_size <= _MAX_MID:
            block_mid = triton.next_power_of_2(mid_size)
            mid = torch.empty((block_mid,), dtype=torch.float32, device=inp.device)
            out = torch.empty([], dtype=dtype, device=inp.device)
            with torch_device_fn.device(inp.device):
                kernel_1_unmasked_pipe[(mid_size,)](
                    inp,
                    target,
                    mid,
                    M,
                    _PIPE_LARGE_BLOCK,
                    _PIPE_CHUNK,
                    reduction,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
                kernel_2[(1, 1, 1)](
                    mid, out, mid_size, block_mid, buffer_size_limit=_BUFFER_SIZE_LIMIT
                )
            return out

    if (M > _FULL_BLOCK) and (M % _FULL_BLOCK == 0):
        # Fully divisible by the largest exact tl.sum tile: the unmasked
        # stage-1 path skips the masked-memory penalty entirely (masked
        # stage-1 tiles at BLOCK >= 32768 run at a fraction of the unmasked
        # bandwidth on XPU). Non-divisible tensors take the legacy path
        # below, which reduces full blocks unmasked and handles the
        # remainder with a single in-bounds window program (kernel_1_tail).
        mid_size = M // _FULL_BLOCK
        if mid_size <= _MAX_MID:
            block_mid = triton.next_power_of_2(mid_size)
            mid = torch.empty((block_mid,), dtype=torch.float32, device=inp.device)
            out = torch.empty([], dtype=dtype, device=inp.device)
            # `mid` is given BLOCK_MID slots: stage 2 reduces the whole tile
            # and drops the padding lanes by value (mid_size is not
            # necessarily a power of two, e.g. [10000, 65536] -> 20000 ->
            # BLOCK_MID 32768).
            with torch_device_fn.device(inp.device):
                kernel_1_unmasked_pipe[(mid_size,)](
                    inp,
                    target,
                    mid,
                    M,
                    _FULL_BLOCK,
                    _PIPE_CHUNK,
                    reduction,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
                kernel_2[(1, 1, 1)](
                    mid, out, mid_size, block_mid, buffer_size_limit=_BUFFER_SIZE_LIMIT
                )
            return out

    if (M > _V2_BLOCK) and (M % _V2_BLOCK == 0):
        # Divisible by 32768 but not by 131072 (e.g. 65536 = 2 tiles): keep
        # the smaller unmasked tile, which beats one 65536-lane program.
        mid_size = M // _V2_BLOCK
        if mid_size <= _MAX_MID:
            block_mid = triton.next_power_of_2(mid_size)
            mid = torch.empty((block_mid,), dtype=torch.float32, device=inp.device)
            out = torch.empty([], dtype=dtype, device=inp.device)
            with torch_device_fn.device(inp.device):
                kernel_1_unmasked_v2[(mid_size,)](
                    inp,
                    target,
                    mid,
                    M,
                    _V2_BLOCK,
                    reduction,
                    buffer_size_limit=_BUFFER_SIZE_LIMIT,
                )
                kernel_2[(1, 1, 1)](
                    mid, out, mid_size, block_mid, buffer_size_limit=_BUFFER_SIZE_LIMIT
                )
            return out

    # Legacy path: stage-1 blocks sized by get_block_size_1d (or the measured
    # 65536-lane optimum for the big non-divisible shapes), fp32 mid
    # accumulation. Stage 1 is fully unmasked: full blocks via kernel_1 and
    # one in-bounds window program for the remainder (kernel_1_tail); the
    # mid padding lanes are dropped by value in kernel_2.
    if M > _LEGACY_BLOCK:
        block_size = _LEGACY_BLOCK
    mid_size = triton.cdiv(M, block_size)
    if mid_size > _MAX_MID:
        block_size = triton.next_power_of_2(triton.cdiv(M, _MAX_MID))
        mid_size = triton.cdiv(M, block_size)
    n_full = M // block_size
    rem = M - n_full * block_size
    mid_size = n_full + (1 if rem else 0)
    block_mid = triton.next_power_of_2(mid_size)

    # BLOCK_MID slots: stage 2 sums the whole tile and drops the padding
    # lanes by value, so the slots past `mid_size` must be readable.
    mid = torch.empty((block_mid,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        kernel_1[(n_full, 1, 1)](
            inp,
            target,
            mid,
            M,
            block_size,
            reduction,
            buffer_size_limit=_BUFFER_SIZE_LIMIT,
        )
        if rem:
            kernel_1_tail[(1, 1, 1)](
                inp,
                target,
                mid,
                M,
                M - block_size,
                rem,
                n_full,
                block_size,
                reduction,
                buffer_size_limit=_BUFFER_SIZE_LIMIT,
            )
        if mid_size == 1:
            return mid.reshape([]).to(dtype)
        kernel_2[(1, 1, 1)](
            mid, out, mid_size, block_mid, buffer_size_limit=_BUFFER_SIZE_LIMIT
        )

    return out
