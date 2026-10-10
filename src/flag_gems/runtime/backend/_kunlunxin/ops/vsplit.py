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
from typing import List, Union

import torch

logger = logging.getLogger(__name__)


def vsplit(input: torch.Tensor, indices_or_sections: Union[int, List[int]]):
    """Split a tensor along the first axis (row-wise), kunlunxin XPU override.

    ``torch.vsplit`` (== ``torch.tensor_split(..., dim=0)``) is a *view*
    operation: native torch returns tensors that share storage with the input
    (zero data movement, ~3us for any shape).  The generic flag_gems
    implementation instead launches one Triton copy kernel per chunk into
    freshly allocated outputs, which on XPU is ~77 GB/s for large shapes
    and pays 2-4 kernel launches per call, so it can never approach the native
    view latency.

    This override mirrors the ``chunk`` override: each output is a pure
    ``torch.as_strided`` metadata view (same stride, shifted storage offset),
    so ``vsplit`` becomes a zero-copy host-side shape/offset computation just
    like the native op.
    """
    logger.debug("GEMS_KUNLUNXIN VSPLIT")

    if input.ndim < 2:
        raise RuntimeError(
            f"vsplit requires a tensor with 2 or more dimensions, got {input.ndim}"
        )

    dim = 0
    dim_size = input.shape[dim]

    # Resolve the chunk sizes along the row dimension.
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        if n <= 0:
            raise ValueError(f"indices_or_sections must be positive, got {n}")
        # torch.vsplit requires an even division, unlike torch.tensor_split.
        if dim_size % n != 0:
            raise RuntimeError(
                f"torch.vsplit attempted to split along dimension {dim}, but the "
                f"size of the dimension {dim_size} is not divisible by the "
                f"split_size {n}!"
            )
        chunk_sizes = [dim_size // n] * n
    else:
        chunk_sizes = []
        prev = 0
        for idx in indices_or_sections:
            idx = max(0, min(idx, dim_size))
            chunk_sizes.append(idx - prev)
            prev = idx
        chunk_sizes.append(dim_size - prev)

    # Zero-copy views: keep the input's stride and just shift the storage
    # offset by the number of leading rows.  Works for contiguous, transposed
    # (non-contiguous) and empty-chunk (size 0 along dim 0) cases alike.
    stride = input.stride()
    storage_offset = input.storage_offset()
    dim_stride = stride[dim]

    result = []
    start = 0
    for chunk_size in chunk_sizes:
        size = list(input.shape)
        size[dim] = chunk_size
        result.append(
            torch.as_strided(input, size, stride, storage_offset + start * dim_stride)
        )
        start += chunk_size

    return tuple(result)
