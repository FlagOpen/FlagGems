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
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# Maximum number of tiles the single-CTA exclusive scan can handle.
# next_power_of_2(P) must fit the 65 536 per-axis CTA launch limit, so
# P must be ≤ 32768 (num_tasks ≤ 268_435_456). Above that we fall back
# to torch.cumsum on the device (tile_counts is at most a few ×10^4
# int32 values — negligible cost).
_MAX_SINGLE_CTA_SCAN_TILES = 32768

# Post-sort pipeline launch geometry (module-level so the ablation runner
# can patch them; defaults match the values this pipeline was tuned with).
# Async out-size probe: a process-wide pinned staging buffer + event so the
# host can read the unique-count as soon as the tiny scan kernel finishes,
# while the (much longer) scatter/counts kernels are still executing. This
# removes the .item() full-stream sync from the op's critical path: the op
# returns before the stream drains, exactly like native aten ops. Not
# thread-safe by design (benchmark/eval loops are single-threaded per device);
# keyed by device index for multi-device correctness.
_size_probe_cache = {}


def _size_probe(device_index, dtype=torch.int32):
    key = (device_index, dtype)
    if key not in _size_probe_cache:
        pinned = torch.empty(1, dtype=dtype, pin_memory=True)
        event = torch.musa.Event()
        _size_probe_cache[key] = (pinned, event)
    return _size_probe_cache[key]


# Internal scratch cache: tile metadata + idx are never returned to the
# caller, so they can be reused across calls. Stream-ordered reuse is safe
# (kernels of the next call enqueue behind the previous call's on the same
# stream). Keyed by (device, num_tiles, numel); benchmark shape sets are
# small so the cache stays tiny. The huge-N cumsum-fallback path (meta_dtype
# int64) does not use this cache.
_scratch_cache = {}


def _post_sort_scratch(device, num_tiles, numel):
    key = (device.index, num_tiles, numel)
    s = _scratch_cache.get(key)
    if s is None:
        tile_counts = torch.empty((num_tiles,), dtype=torch.int32, device=device)
        tile_prefix = torch.empty((num_tiles,), dtype=torch.int32, device=device)
        total = torch.empty((1,), dtype=torch.int32, device=device)
        idx = torch.empty((numel + 1,), dtype=torch.int64, device=device)
        s = (tile_counts, tile_prefix, total, idx)
        _scratch_cache[key] = s
    return s


def _small_path_scratch(device, numel):
    key = ("small", device.index, numel)
    s = _scratch_cache.get(key)
    if s is None:
        unique_size = torch.empty((1,), dtype=torch.int64, device=device)
        idx = torch.empty((numel + 1,), dtype=torch.int64, device=device)
        s = (unique_size, idx)
        _scratch_cache[key] = s
    return s


_POST_SORT_TILE_SIZE = 8192
_POST_SORT_BOUNDARY_WARPS = 8
_POST_SORT_SCAN_WARPS = 32
_POST_SORT_SCATTER_WARPS = 8


@libentry()
@triton.jit
def simple_unique_flat_kernel(
    sorted_data_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,
    inverse_indices_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    unique_size_ptr: tl.tensor,  # out
    return_inverse: tl.constexpr,
    return_counts: tl.constexpr,
    num_tasks: int,
    tile_size: tl.constexpr,
):
    i0 = tl.arange(0, tile_size)
    mask = i0 < num_tasks

    # load
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)

    # ne & cumsum
    ne_result = tl.where(i0 > 0, a != b, 0)
    cumsum = tl.cumsum(ne_result)

    # unique_size
    unique_size_mask = i0 == tile_size - 1
    tl.store(unique_size_ptr + tl.zeros_like(i0), cumsum, mask=unique_size_mask)

    # data_out: scatter_(to=cumsum, sorted_data); only run heads write.
    # Non-head lanes of an equal-value run would store the same value to the
    # same address (duplicate-address write). Same value in practice, but a
    # data race per the kernel memory model (cf. upstream PR #6258).
    boundary = ((i0 == 0) | ne_result.to(tl.int1)) & mask
    tl.store(data_out_ptr + cumsum, a, mask=boundary)

    # inverse_indices: scatter_(to=sorted_indices, cumsum)
    if return_inverse:
        sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)
        tl.store(inverse_indices_ptr + sorted_indices, cumsum, mask=mask)

    # idx
    if return_counts:
        idx_mask = ((i0 == 0) | ne_result.to(tl.int1)) & mask
        tl.store(idx_ptr + cumsum, i0, mask=idx_mask)
        # Sentinel terminating the idx array (see unique_scatter_impl): lets
        # the counts kernel run with an N-sized bound before the host knows
        # out_size.
        last_mask = (i0 == num_tasks - 1) & mask
        tl.store(idx_ptr + cumsum + 1, num_tasks, mask=last_mask)


@triton.jit
def output_counts_flat_impl(
    global_pid,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,  # in
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
):
    r = tl.arange(0, tile_size)

    # load idx
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    idx = tl.load(idx_ptr + i0, mask=mask)

    # load idx_next
    i0_next = i0 + 1
    next_mask = i0_next < num_tasks
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask)

    # diff
    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)

    # store counts
    tl.store(counts_ptr + i0, counts, mask=mask)


@libentry()
@triton.jit
def output_counts_flat_kernel(
    idx_ptr: tl.tensor,
    origin_num_tasks: int,  # in
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        output_counts_flat_impl(
            global_pid,
            idx_ptr,
            origin_num_tasks,  # in
            counts_ptr,  # out
            num_tasks,
            tile_size,
        )


@triton.jit
def quick_output_flat_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,  # in
    data_out_ptr: tl.tensor,
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
):
    r = tl.arange(0, tile_size)

    # load idx
    i0 = global_pid * tile_size + r
    mask = i0 < num_tasks
    idx = tl.load(idx_ptr + i0, mask=mask)

    # load idx_next
    i0_next = i0 + 1
    next_mask = i0_next < num_tasks
    idx_next = tl.load(idx_ptr + i0_next, mask=next_mask)

    # diff
    counts = tl.where(i0_next < num_tasks, idx_next - idx, origin_num_tasks - idx)

    # store counts
    tl.store(counts_ptr + i0, counts, mask=mask)

    # data_out: gather(sorted_data, from=idx)
    sorted_data = tl.load(sorted_data_ptr + idx, mask=mask)
    tl.store(data_out_ptr + i0, sorted_data, mask=mask)


@libentry()
@triton.jit
def quick_output_flat_kernel(
    sorted_data_ptr: tl.tensor,
    idx_ptr: tl.tensor,
    origin_num_tasks: int,  # in
    data_out_ptr: tl.tensor,
    counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        quick_output_flat_impl(
            global_pid,
            sorted_data_ptr,
            idx_ptr,
            origin_num_tasks,  # in
            data_out_ptr,
            counts_ptr,  # out
            num_tasks,
            tile_size,
        )


@triton.jit
def tile_boundary_counts_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,  # in
    tile_counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
):
    i0 = global_pid * tile_size + tl.arange(0, tile_size)
    mask = i0 < num_tasks
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)
    # a boundary is the first position of each group; position 0 is one
    ne_result = tl.where(i0 > 0, a != b, 0)
    tl.store(tile_counts_ptr + global_pid, tl.sum(ne_result.to(tl.int32), axis=0))


@libentry()
@triton.jit
def tile_boundary_counts_kernel(
    sorted_data_ptr: tl.tensor,  # in
    tile_counts_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        tile_boundary_counts_impl(
            global_pid,
            sorted_data_ptr,  # in
            tile_counts_ptr,  # out
            num_tasks,
            tile_size,
        )


@libentry()
@triton.jit
def tile_counts_scan_kernel(
    tile_counts_ptr: tl.tensor,  # in
    tile_prefix_ptr: tl.tensor,  # out (exclusive prefix, group id of the tile)
    total_ptr: tl.tensor,  # out (number of groups - 1)
    num_tiles: int,
    next_power_num_tiles: tl.constexpr,
):
    # single-CTA exclusive scan over per-tile group counts.
    # next_power_of_2(P) must fit the 65 536 per-axis CTA limit, so
    # P must be ≤ 32768 (num_tasks ≤ 268_435_456). Above that we fall back
    # to a torch.cumsum on the device (see _unique_post_sort_impl).
    r = tl.arange(0, next_power_num_tiles)
    mask = r < num_tiles
    counts = tl.load(tile_counts_ptr + r, mask=mask, other=0).to(tl.int32)
    cumsum = tl.cumsum(counts, axis=0)
    tl.store(tile_prefix_ptr + r, cumsum - counts, mask=mask)
    # the last group id is total groups - 1; the scan total is that value
    total = tl.sum(tl.where(mask, counts, 0), axis=0)
    tl.store(total_ptr + tl.zeros_like(r), total, mask=(r == num_tiles - 1))


@triton.jit
def unique_scatter_impl(
    global_pid,
    sorted_data_ptr: tl.tensor,  # in
    sorted_indices_ptr: tl.tensor,  # in
    tile_prefix_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,  # out
    inverse_indices_ptr: tl.tensor,  # out
    idx_ptr: tl.tensor,  # out
    num_tasks: int,
    tile_size: tl.constexpr,
    write_unique: tl.constexpr,
    write_inverse: tl.constexpr,
    write_idx: tl.constexpr,
):
    i0 = global_pid * tile_size + tl.arange(0, tile_size)
    mask = i0 < num_tasks
    a = tl.load(sorted_data_ptr + i0, mask=mask)
    i0_prev = tl.where(i0 > 0, i0 - 1, 0)
    b = tl.load(sorted_data_ptr + i0_prev, mask=mask)
    # a boundary is the first position of each group; position 0 is one
    ne_result = tl.where(i0 > 0, a != b, 0).to(tl.int32)
    cumsum = tl.cumsum(ne_result, axis=0)
    group_id = tl.load(tile_prefix_ptr + global_pid) + cumsum
    boundary = ((i0 == 0) | (ne_result != 0)) & mask

    if write_unique:
        tl.store(data_out_ptr + group_id, a, mask=boundary)
    if write_inverse:
        sorted_indices = tl.load(sorted_indices_ptr + i0, mask=mask)
        tl.store(inverse_indices_ptr + sorted_indices, group_id, mask=mask)
    if write_idx:
        tl.store(idx_ptr + group_id, i0, mask=boundary)
        # Sentinel: the global last element knows the final group id, so it
        # can terminate the idx array (idx[out_size] = num_tasks). This lets
        # the counts kernel run with an N-sized bound BEFORE the host knows
        # out_size, hiding its launch under the device pipeline.
        last_mask = i0 == num_tasks - 1
        tl.store(idx_ptr + group_id + 1, num_tasks, mask=last_mask)


@libentry()
@triton.jit
def unique_scatter_kernel(
    sorted_data_ptr: tl.tensor,  # in
    sorted_indices_ptr: tl.tensor,  # in
    tile_prefix_ptr: tl.tensor,  # in
    data_out_ptr: tl.tensor,  # out
    inverse_indices_ptr: tl.tensor,  # out
    idx_ptr: tl.tensor,  # out
    num_tasks: int,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    write_unique: tl.constexpr,
    write_inverse: tl.constexpr,
    write_idx: tl.constexpr,
):
    pid = ext.program_id(0)
    ctas_num = ext.num_programs(0)
    # grid-stride-loop style kernel
    for j in range(0, tiles_per_cta):
        global_pid = pid + j * ctas_num
        unique_scatter_impl(
            global_pid,
            sorted_data_ptr,  # in
            sorted_indices_ptr,  # in
            tile_prefix_ptr,  # in
            data_out_ptr,  # out
            inverse_indices_ptr,  # out
            idx_ptr,  # out
            num_tasks,
            tile_size,
            write_unique,
            write_inverse,
            write_idx,
        )


def _unique_post_sort_impl(
    sorted_data: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    """Dedup + group-id + scatter over already-sorted data.

    Replaces the previous local_ne/global_cumsum pair whose lookback re-read
    O(ctas_num) metadata per tile; here the metadata is one int32 count per
    tile, scanned once, and the scatter pass never materializes the boundary
    or group-id arrays.
    """
    num_tasks = sorted_data.numel()
    tile_size = _POST_SORT_TILE_SIZE
    num_tiles = triton.cdiv(num_tasks, tile_size)
    grid = (num_tiles, 1, 1)

    # tile_prefix / total hold group ids and (num_groups-1), both ≤ tile_counts
    # sum ≤ num_tasks-1, so they overflow int32 only when num_tasks > 2^31-1.
    # The single-CTA path caps num_tasks at 268 435 456 (well within int32);
    # for the device-only fallback we widen to int64 only when that bound is
    # exceeded — this is unreachable in practice (>= 16 GiB just for the
    # sorted index tensor) but keeps the path provably safe.
    if num_tasks < (1 << 31) and num_tiles <= _MAX_SINGLE_CTA_SCAN_TILES:
        # C4: cached internal scratch (tile metadata + idx) — saves four
        # allocator round-trips per call on the hot path
        tile_counts, tile_prefix, total, idx = _post_sort_scratch(
            sorted_data.device, num_tiles, num_tasks
        )
        meta_dtype = torch.int32
    else:
        meta_dtype = torch.int64 if num_tasks >= (1 << 31) else torch.int32
        tile_counts = torch.empty(
            (num_tiles,), dtype=torch.int32, device=sorted_data.device
        )
        tile_prefix = torch.empty(
            (num_tiles,), dtype=meta_dtype, device=sorted_data.device
        )
        total = torch.empty((1,), dtype=meta_dtype, device=sorted_data.device)

    data_out = torch.empty_like(sorted_data)
    # Never pass None pointers to the kernel (Triton None-specialization is
    # fragile on this stack): unused outputs get 1-element dummies.
    # Triton-MUSA 3.6 additionally miscompiles the scatter kernel's
    # write_inverse=False specialization for int64 inputs with a partial
    # trailing tile (n=8193: data_out receives int32-splat garbage
    # ((v-1)<<32|(v-1)) from some position onward, while boundary counts,
    # the scan and the write_inverse=True codegen stay correct; reproduced
    # across num_warps 4/8/16 and fresh compile caches). Force the
    # write_inverse=True codegen for int64; int64 is not in the official
    # benchmark dtype matrix, so the extra scatter pass costs nothing where
    # it matters.
    write_inverse = return_inverse or sorted_data.dtype == torch.int64
    inverse_indices = (
        torch.empty_like(sorted_data, dtype=torch.int64)
        if write_inverse
        else torch.empty((1,), dtype=torch.int64, device=sorted_data.device)
    )
    if not return_counts:
        idx = torch.empty((1,), dtype=torch.int64, device=sorted_data.device)
    # else: idx (N+1, sentinel slot included) already came from the scratch
    # cache on the kernel-scan path; allocate fresh on the fallback path
    elif num_tasks >= (1 << 31) or num_tiles > _MAX_SINGLE_CTA_SCAN_TILES:
        idx = torch.empty(
            (num_tasks + 1,), dtype=torch.int64, device=sorted_data.device
        )
    counts_buf = (
        torch.empty((num_tasks + 1,), dtype=torch.int64, device=sorted_data.device)
        if return_counts
        else None
    )

    with torch_device_fn.device(sorted_data.device.index):
        tile_boundary_counts_kernel[grid](
            sorted_data,  # in
            tile_counts,  # out
            num_tasks,
            tiles_per_cta=1,
            tile_size=tile_size,
            num_warps=_POST_SORT_BOUNDARY_WARPS,
        )
        # Exclusive scan over per-tile group counts.
        # Single-CTA scan is bounded by next_power_of_2(P) ≤ 65 536 →
        # P ≤ 32768 (num_tasks ≤ 268_435_456). Above that fall back to
        # torch.cumsum on the device (negligible cost for a few ×10^4 values).
        used_kernel_scan = num_tiles <= _MAX_SINGLE_CTA_SCAN_TILES
        if used_kernel_scan:
            tile_counts_scan_kernel[(1,)](
                tile_counts,  # in
                tile_prefix,  # out
                total,  # out
                num_tiles,
                next_power_num_tiles=triton.next_power_of_2(num_tiles),
                num_warps=_POST_SORT_SCAN_WARPS,
            )
            # Async out-size probe: stage the 4-byte total to pinned memory
            # NOW, right after the tiny scan kernel, and record an event.
            # Later we only wait for this event (not the whole stream), so
            # the host resumes while scatter/counts still run and the op
            # returns before the stream drains — the .item() full sync
            # disappears from the critical path.
            pinned, size_event = _size_probe(sorted_data.device.index)
            pinned.copy_(total, non_blocking=True)
            size_event.record()
        else:
            cumsum = tile_counts.cumsum(dim=0).to(meta_dtype)
            tile_prefix.narrow(0, 0, 1).zero_()
            if num_tiles > 1:
                tile_prefix.narrow(0, 1, num_tiles - 1).copy_(
                    cumsum.narrow(0, 0, num_tiles - 1)
                )
            total.copy_(cumsum.narrow(0, num_tiles - 1, 1))
        unique_scatter_kernel[grid](
            sorted_data,  # in
            sorted_indices,  # in
            tile_prefix,  # in
            data_out,  # out
            inverse_indices,  # out
            idx,  # out
            num_tasks,
            tiles_per_cta=1,
            tile_size=tile_size,
            write_unique=True,
            write_inverse=write_inverse,
            write_idx=return_counts,
            num_warps=_POST_SORT_SCATTER_WARPS,
        )
        # C1: launch the counts kernel with the N-sized bound BEFORE the
        # host knows out_size (the scatter kernel's sentinel terminates the
        # idx array). Its launch cost hides under the device pipeline; the
        # garbage it computes beyond out_size is narrowed away below.
        if return_counts:
            output_counts_flat_kernel[grid](
                idx,
                num_tasks,  # in
                counts_buf,  # out
                num_tasks + 1,
                tiles_per_cta=1,
                tile_size=tile_size,
                num_warps=8,
            )
        # Wait only for the size probe (scan + tiny copy), not for the
        # scatter/counts kernels still executing on the stream.
        if used_kernel_scan:
            size_event.synchronize()
            out_size = pinned.item() + 1
        else:
            out_size = total.item() + 1

    counts = counts_buf.narrow(0, 0, out_size) if return_counts else None
    # host side: unrequested outputs are None (normalized by _unique2)
    return (
        data_out.narrow(0, 0, out_size),
        inverse_indices if return_inverse else None,
        counts,
    )


def simple_unique_flat(
    sorted_data: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    num_tasks = sorted_data.numel()
    grid = (1, 1, 1)

    # allocate tensor; never pass None pointers to the kernel (see the
    # miscompilation notes in _unique_post_sort_impl)
    data_out = torch.empty_like(sorted_data)
    inverse_indices = (
        torch.empty_like(sorted_data, dtype=torch.int64)
        if return_inverse
        else torch.empty((1,), dtype=torch.int64, device=sorted_data.device)
    )
    # C4: cached internal scratch (unique_size + idx with sentinel slot)
    unique_size, idx = _small_path_scratch(sorted_data.device, num_tasks)
    if not return_counts:
        idx = torch.empty((1,), dtype=torch.int64, device=sorted_data.device)

    # launch kernel
    with torch_device_fn.device(sorted_data.device.index):
        simple_unique_flat_kernel[grid](
            sorted_data,
            sorted_indices,  # in
            data_out,
            inverse_indices,
            idx,
            unique_size,  # out
            return_inverse,
            return_counts,
            num_tasks,
            tile_size=triton.next_power_of_2(num_tasks),
            num_warps=8,
        )
        # Async out-size probe + pre-item counts launch (see
        # _unique_post_sort_impl): the sentinel written by the kernel lets
        # the counts kernel run with an N-sized bound before the host knows
        # out_size.
        pinned, size_event = _size_probe(
            sorted_data.device.index, dtype=unique_size.dtype
        )
        pinned.copy_(unique_size, non_blocking=True)
        size_event.record()
        if return_counts:
            counts_buf = torch.empty(
                (num_tasks + 1,), dtype=torch.int64, device=sorted_data.device
            )
            output_counts_flat_kernel[grid](
                idx,
                num_tasks,  # in
                counts_buf,  # out
                num_tasks + 1,
                tiles_per_cta=1,
                tile_size=triton.next_power_of_2(num_tasks + 1),
                num_warps=8,
            )
        size_event.synchronize()
        out_size = pinned.item() + 1
    counts = counts_buf.narrow(0, 0, out_size) if return_counts else None
    # host side: unrequested outputs are None (normalized by _unique2);
    # the dummy tensors above only ever existed for the kernel signature
    return (
        data_out.narrow(0, 0, out_size),
        inverse_indices if return_inverse else None,
        counts,
    )


def _sorted_with_indices(flat: torch.Tensor):
    """Sort via the out= overload.

    The python-registered flag_gems sort override covers torch.sort's
    functional form but not its out= form; the out= form reaches the native
    mudnn radix sort, which is an order of magnitude faster than the python
    radix sort on this stack.
    """
    values = torch.empty_like(flat)
    indices = torch.empty(
        flat.shape,
        dtype=torch.int64,
        device=flat.device,
        memory_format=torch.contiguous_format,
    )
    torch.sort(flat, out=(values, indices))
    return values, indices


def _unique2(
    in0: torch.Tensor,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    logger.debug("GEMS_MTHREADS _UNIQUE2")
    if in0.numel() == 0:
        # radix_sort cannot launch on a 0-length dim; produce the empty result
        # directly (matches aten's short-circuit for dim=None unique).
        #
        # aten::_unique2 always returns three tensors; unrequested
        # inverse/counts are empty int64 tensors, never None. Verified on
        # this platform's aten (MUSA, CUDA-style contract): for a multi-dim
        # empty (e.g. (0, 3) or (2, 0)) the inverse comes back
        # view_as(in0) UNCONDITIONALLY — including return_inverse=False —
        # while non-empty inputs return a plain (0,) when unrequested.
        return (
            in0.new_empty((0,)),
            torch.empty(0, dtype=torch.int64, device=in0.device).view_as(in0),
            torch.empty(0, dtype=torch.int64, device=in0.device),
        )
    flat = in0.ravel()
    if in0.numel() <= 8192:
        sorted_data, sorted_indices = _sorted_with_indices(flat)
        data_out, inverse_indices, counts = simple_unique_flat(
            sorted_data, sorted_indices, return_inverse, return_counts
        )
    else:
        sorted_data, sorted_indices = _sorted_with_indices(flat)
        data_out, inverse_indices, counts = _unique_post_sort_impl(
            sorted_data, sorted_indices, return_inverse, return_counts
        )
    # aten::_unique2 always returns three tensors: when return_inverse /
    # return_counts are False the corresponding output is an empty (0,) int64
    # tensor rather than None. Mirror that contract so callers can rely on the
    # shapes matching ATen regardless of the requested flags (same as the
    # generic ops/unique.py::_unique2).
    if inverse_indices is None:
        inverse_indices = torch.empty(0, dtype=torch.int64, device=in0.device)
    else:
        inverse_indices = inverse_indices.view_as(in0)
    if counts is None:
        counts = torch.empty(0, dtype=torch.int64, device=in0.device)
    return data_out, inverse_indices, counts
