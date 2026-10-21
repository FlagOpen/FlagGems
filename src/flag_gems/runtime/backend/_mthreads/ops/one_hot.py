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
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Dense generation speculatively allocates and writes the output before
# reporting an invalid index.  S5000 crossover measurements show the dense
# kernel beating zero-fill + scatter through ~64M output elements (the fill
# plus the scattered 8-byte stores amplify DRAM traffic ~16x), so keep the
# speculative allocation at or below 512 MiB.
_DENSE_MAX_OUTPUT_ELEMENTS = 67_108_864

# Beyond this class count the per-row comparison loop inside one program gets
# long enough that zero-filling the output and scattering only the "1"
# positions becomes cheaper than computing every element.
_DENSE_MAX_CLASSES = 8192


# ── dense generation kernels (single pass over rows, tiled over classes) ─────


@libentry()
@triton.jit
def one_hot_dense_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
    CLASS_BLOCK: tl.constexpr,
):
    # 32-bit ids keep the offset math in i32, which avoids keeping per-row
    # base products alive as i64 register pairs across the class loop (193
    # vs 107 registers at CLASS_BLOCK=128) — for class counts not divisible
    # by 16 that extra register pressure hits a superlinear occupancy cliff
    # (measured up to 9.4x slower).  Safe because both dense paths are
    # guarded to outputs below 2**31 elements.
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < num_elements
    indices = tl.load(input_ptr + rows, mask=row_mask, other=0)

    cols = tl.arange(0, CLASS_BLOCK)
    for col_start in range(0, num_classes, CLASS_BLOCK):
        class_offsets = col_start + cols
        col_mask = class_offsets < num_classes
        values = tl.where(indices[:, None] == class_offsets[None, :], 1, 0)
        values = values.to(tl.int64)
        out_offsets = rows[:, None] * num_classes + class_offsets[None, :]
        tl.store(
            output_ptr + out_offsets, values, mask=row_mask[:, None] & col_mask[None, :]
        )


@libentry()
@triton.jit
def one_hot_validate_dense_kernel(
    input_ptr,
    output_ptr,
    status_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
    CLASS_BLOCK: tl.constexpr,
):
    """Write the full one-hot output and record a per-block error code.

    Status values preserve the existing error precedence: negative indices
    take priority over indices greater than or equal to ``num_classes``.
    Every program owns its own status slot, so the buffer needs no
    initialization and the kernel needs no atomics.
    """
    # 32-bit ids: see one_hot_dense_kernel.
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < num_elements
    indices = tl.load(input_ptr + rows, mask=row_mask, other=0)

    negative = indices < 0
    too_large = indices >= num_classes
    code = tl.where(negative, 2, tl.where(too_large, 1, 0))
    block_code = tl.max(tl.where(row_mask, code, 0), axis=0)
    tl.store(status_ptr + pid, block_code.to(tl.int32))

    cols = tl.arange(0, CLASS_BLOCK)
    for col_start in range(0, num_classes, CLASS_BLOCK):
        class_offsets = col_start + cols
        col_mask = class_offsets < num_classes
        values = tl.where(indices[:, None] == class_offsets[None, :], 1, 0)
        values = values.to(tl.int64)
        out_offsets = rows[:, None] * num_classes + class_offsets[None, :]
        tl.store(
            output_ptr + out_offsets, values, mask=row_mask[:, None] & col_mask[None, :]
        )


# ── fused min/max reduction (num_classes inference) ──────────────────────────


@libentry()
@triton.jit
def one_hot_status_reduce_kernel(
    status_ptr,
    out_ptr,
    num_blocks,
    BLOCK: tl.constexpr,
):
    """Reduce per-block status slots to a single code on the device.

    Returning the verdict through a 4-byte readback keeps the host-side
    blocking copy on the same tiny-transfer driver path as the reference
    implementation, which measures stable across processes; multi-KiB
    pageable readbacks are bimodal (fast/slow driver staging modes).
    """
    code = tl.full((), 0, tl.int32)
    for start in range(0, num_blocks, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < num_blocks
        values = tl.load(status_ptr + offs, mask=mask, other=0)
        code = tl.maximum(code, tl.max(values, axis=0))
    tl.store(out_ptr, code)


@libentry()
@triton.jit
def one_hot_minmax_kernel(
    input_ptr,
    status_ptr,
    num_elements,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute per-block min/max so the host can infer ``num_classes``.

    Block minima land in ``status[:num_blocks]`` and block maxima in
    ``status[num_blocks:]``; the host performs one readback and two tiny CPU
    reductions, replacing the previous aminmax + two ``.item()`` syncs with a
    single synchronization.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    block_min = tl.min(tl.where(mask, indices, 1 << 62), axis=0)
    block_max = tl.max(tl.where(mask, indices, -(1 << 62)), axis=0)
    tl.store(status_ptr + pid, block_min)
    tl.store(status_ptr + num_blocks + pid, block_max)


# ── scatter kernels: only write the "1" positions (output zeroed first) ──────


@libentry()
@triton.jit
def one_hot_set_one_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_offsets = offsets * num_classes + indices
    tl.store(output_ptr + out_offsets, 1, mask=mask)


@libentry()
@triton.jit
def one_hot_validate_set_one_kernel(
    input_ptr,
    output_ptr,
    status_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    """Validate indices while scattering into a zero-initialized output.

    Status values preserve the existing error precedence: negative indices
    take priority over indices greater than or equal to ``num_classes``.
    Every program owns its own status slot, so the buffer needs no
    initialization and the kernel needs no atomics.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    negative = indices < 0
    too_large = indices >= num_classes
    valid = mask & ~negative & ~too_large
    safe_indices = tl.minimum(tl.maximum(indices, 0), num_classes - 1)
    tl.store(
        output_ptr + offsets * num_classes + safe_indices,
        1,
        mask=valid,
    )
    code = tl.where(negative, 2, tl.where(too_large, 1, 0))
    block_code = tl.max(tl.where(mask, code, 0), axis=0)
    tl.store(status_ptr + pid, block_code.to(tl.int32))


# ── launch configuration helpers ──────────────────────────────────────────────


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _dense_config(num_elements: int, num_classes: int):
    """Pick row-block, class-tile and warp sizes for the dense kernels.

    The per-program tile is capped at 4096 int64 values (32 KiB): S5000
    crossover measurements show wide row blocks (256 rows at 16 classes)
    writing larger contiguous spans beat smaller tiles by ~7% on
    bandwidth-bound outputs.  Row blocks are additionally sized so the
    per-block status buffer lands in a fast pageable-DtoH band: either tiny
    (<= 128 B) for small inputs, or 2 KiB and above for larger ones — S5000
    pageable copies in the 256 B-1 KiB range run through serialized
    ~512 B staging chunks and cost ~20 us.
    """
    class_block = min(max(_next_pow2(num_classes), 16), 256)
    hi = max(4, min(256, 4096 // class_block))
    if num_elements <= 2048:
        rows = _next_pow2(max(1, num_elements // 32))
    else:
        rows = _next_pow2(max(1, num_elements // 1024))
    rows = max(4, min(rows, hi))
    return rows, class_block, 4


def _minmax_block_size(num_elements: int):
    """Pick a BLOCK_SIZE for the fused min/max kernel.

    The status buffer holds 16 bytes per block; steer its total size away
    from the slow 256 B-1 KiB pageable-DtoH band (see ``_dense_config``).
    """
    for block in (1024, 2048, 4096, 8192):
        size = 16 * triton.cdiv(num_elements, block)
        if size <= 256 or size >= 2048:
            return block
    return 1024


def _scatter_block_size(num_elements: int) -> int:
    """Pick a BLOCK_SIZE for the scatter-only kernel."""
    if num_elements <= 1024:
        return 256
    elif num_elements <= 16384:
        return 512
    else:
        return 1024


# Torch CPU reductions fork an OpenMP team above this grain size, which costs
# ~10 ms per call on large hosts — far more than the entire one_hot call.  The
# per-block status buffers can exceed it for very large inputs, so host-side
# reductions over them are chunked to stay on the serial fast path.
_HOST_REDUCTION_GRAIN = 32768


def _host_max(values: torch.Tensor) -> int:
    n = values.numel()
    if n <= _HOST_REDUCTION_GRAIN:
        return int(values.max())
    result = None
    for i in range(0, n, _HOST_REDUCTION_GRAIN):
        chunk = int(values[i : i + _HOST_REDUCTION_GRAIN].max())
        if result is None or chunk > result:
            result = chunk
    return result


def _host_min(values: torch.Tensor) -> int:
    n = values.numel()
    if n <= _HOST_REDUCTION_GRAIN:
        return int(values.min())
    result = None
    for i in range(0, n, _HOST_REDUCTION_GRAIN):
        chunk = int(values[i : i + _HOST_REDUCTION_GRAIN].min())
        if result is None or chunk < result:
            result = chunk
    return result


# ── main entry point ──────────────────────────────────────────────────────────


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    logger.debug("GEMS_MTHREADS ONE_HOT")

    if tensor.dtype != torch.int64:
        raise RuntimeError(
            "one_hot is only applicable to index tensor of type LongTensor."
        )

    if tensor.numel() == 0:
        if num_classes <= 0:
            raise RuntimeError(
                "Can not infer total number of classes from empty tensor."
            )
        shape = (*tensor.shape, num_classes)
        return torch.empty(shape, device=tensor.device, dtype=torch.int64)

    inferred_num_classes = num_classes == -1
    on_device = tensor.device.type != "cpu"

    if not on_device:
        if inferred_num_classes:
            num_classes = int(tensor.max().item()) + 1
            if (tensor < 0).any():
                raise RuntimeError("Class values must be non-negative.")
        else:
            invalid = (tensor < 0) | (tensor >= num_classes)
            if invalid.any():
                if (tensor < 0).any():
                    raise RuntimeError("Class values must be non-negative.")
                else:
                    raise RuntimeError("Class values must be smaller than num_classes.")
        if num_classes < 1:
            raise RuntimeError("num_classes should be positive")
        out = torch.zeros((*tensor.shape, num_classes), device="cpu", dtype=torch.int64)
        out.scatter_(-1, tensor.unsqueeze(-1), 1)
        return out

    # An explicit positive num_classes validates on device with a single
    # fused kernel instead of multi-kernel host-side checks.
    device_explicit = (not inferred_num_classes) and num_classes > 0

    flat_input = tensor.contiguous().view(-1)
    num_elements = flat_input.numel()

    with torch_device_fn.device(tensor.device):
        if inferred_num_classes:
            # One fused reduction + one readback replaces aminmax plus two
            # .item() synchronizations.
            BLOCK_SIZE = _minmax_block_size(num_elements)
            num_warps = max(4, min(8, BLOCK_SIZE // 1024))
            num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
            stat = torch.empty(2 * num_blocks, device=tensor.device, dtype=torch.int64)
            one_hot_minmax_kernel[(num_blocks,)](
                flat_input,
                stat,
                num_elements,
                num_blocks,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
            stat_host = torch.empty(2 * num_blocks, dtype=torch.int64)
            stat_host.copy_(stat)
            min_val = _host_min(stat_host[:num_blocks])
            max_val = _host_max(stat_host[num_blocks:])
            if min_val < 0:
                raise RuntimeError("Class values must be non-negative.")
            num_classes = max_val + 1
        elif not device_explicit:
            # A num_classes that is neither -1 nor positive keeps the
            # reference host-side validation semantics on device tensors.
            invalid = (tensor < 0) | (tensor >= num_classes)
            if invalid.any():
                if (tensor < 0).any():
                    raise RuntimeError("Class values must be non-negative.")
                else:
                    raise RuntimeError("Class values must be smaller than num_classes.")

        if num_classes < 1:
            raise RuntimeError("num_classes should be positive")

        if device_explicit:
            if (
                num_elements * num_classes <= _DENSE_MAX_OUTPUT_ELEMENTS
                and num_classes <= _DENSE_MAX_CLASSES
            ):
                # Single pass: write every output element and the per-block
                # error codes, then one readback decides whether to raise.
                rows, class_block, num_warps = _dense_config(num_elements, num_classes)
                num_blocks = triton.cdiv(num_elements, rows)
                out = torch.empty(
                    num_elements * num_classes,
                    device=tensor.device,
                    dtype=torch.int64,
                )
                status = torch.empty(
                    num_blocks, device=tensor.device, dtype=torch.int32
                )
                one_hot_validate_dense_kernel[(num_blocks,)](
                    flat_input,
                    out,
                    status,
                    num_elements,
                    num_classes,
                    BLOCK_SIZE=rows,
                    CLASS_BLOCK=class_block,
                    num_warps=num_warps,
                )
                error_code = torch.empty(1, device=tensor.device, dtype=torch.int32)
                one_hot_status_reduce_kernel[(1,)](
                    status,
                    error_code,
                    num_blocks,
                    BLOCK=1024,
                )
                error = int(error_code.item())
            else:
                out = torch.zeros(
                    num_elements * num_classes,
                    device=tensor.device,
                    dtype=torch.int64,
                )
                BLOCK_SIZE = _scatter_block_size(num_elements)
                num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
                status = torch.empty(
                    num_blocks, device=tensor.device, dtype=torch.int32
                )
                one_hot_validate_set_one_kernel[(num_blocks,)](
                    flat_input,
                    out,
                    status,
                    num_elements,
                    num_classes,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                status_host = torch.empty(num_blocks, dtype=torch.int32)
                status_host.copy_(status)
                error = _host_max(status_host)
            if error == 2:
                raise RuntimeError("Class values must be non-negative.")
            if error == 1:
                raise RuntimeError("Class values must be smaller than num_classes.")
        else:
            # Inferred classes are valid by construction (0 <= v <= max), so
            # generation needs no validation pass at all.  The same output
            # budget as the explicit dense path keeps the kernels' 32-bit
            # offset math safe; larger inferred outputs fall back to the
            # 64-bit scatter path.
            if (
                num_classes <= _DENSE_MAX_CLASSES
                and num_elements * num_classes <= _DENSE_MAX_OUTPUT_ELEMENTS
            ):
                rows, class_block, num_warps = _dense_config(num_elements, num_classes)
                out = torch.empty(
                    num_elements * num_classes,
                    device=tensor.device,
                    dtype=torch.int64,
                )
                one_hot_dense_kernel[(triton.cdiv(num_elements, rows),)](
                    flat_input,
                    out,
                    num_elements,
                    num_classes,
                    BLOCK_SIZE=rows,
                    CLASS_BLOCK=class_block,
                    num_warps=num_warps,
                )
            else:
                out = torch.zeros(
                    num_elements * num_classes,
                    device=tensor.device,
                    dtype=torch.int64,
                )
                BLOCK_SIZE = _scatter_block_size(num_elements)
                one_hot_set_one_kernel[(triton.cdiv(num_elements, BLOCK_SIZE),)](
                    flat_input,
                    out,
                    num_elements,
                    num_classes,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

    return out.view(*tensor.shape, num_classes)
