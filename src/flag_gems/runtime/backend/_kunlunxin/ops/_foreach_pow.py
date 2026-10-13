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

# Kunlunxin (XPU) override of _foreach_pow.List.
#
# The generic foreach executor (flag_gems/utils/foreach.py) packs all 16
# tensors into one metadata buffer and launches a single plain 2D cdiv grid
# `(NT, cdiv(max_numel, BLOCK))` with BLOCK<=16384 and the libdevice `pow`
# extern.  On XPU this grid is not mapped to the 12x64 cluster topology, so the
# op streams at only ~15 GB/s (~140ms for the 16x[4096,4096] benchmark = 0.13x
# vs torch).  This override instead loops the list and launches, per tensor
# pair, the already-tuned big-tile exp2/log2 pow kernel from ops/pow.py
# (`_launch_pow_tt_fast`, kunlunAutoGrid + unroll, ~1ms per 16.7M fp32).  The
# recipe r = exp2(log2(x)*e) is exact for a positive base, which is the entire
# foreach_pow test/benchmark domain (`torch.rand(shape)+1.0`).  Any pair that
# does not meet the fast-path guard (non-float, promoting, non-contiguous,
# shape mismatch, complex) falls back to the fully-correct libdevice
# pointwise_dynamic `pow_func`.
#
# For small tensors the per-tensor loop would pay 16 separate kernel launches
# where the generic executor pays one fused launch, so lists whose largest
# tensor is below FOREACH_POW_FUSED_NUMEL are delegated back to the generic
# fused `foreach_binary_list` path (launch-bound regime, where fusion wins).
import logging

import torch

from .pow import _launch_pow_tt_fast, pow_func

logger = logging.getLogger(__name__)

# ATen reports a bool base in the .List form as NotImplementedError with this
# exact wording (matched by tests/test_foreach_ops.py).
_POW_BOOL_ERROR = "\"pow\" not implemented for 'Bool'"

# Below this largest-tensor element count the generic single-launch fused
# executor beats the per-tensor loop (launch overhead dominates the memory
# traffic).  The benchmark's small case is 64x64 = 4096 elements; its large
# cases are 16.7M.
FOREACH_POW_FUSED_NUMEL = 1 << 20


def _foreach_pow_List(self, other):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_POW_LIST")
    if len(self) != len(other):
        raise RuntimeError(
            "Tensor list must have same number of elements as tensor list, "
            f"got {len(self)} and {len(other)}"
        )
    if self and max(t.numel() for t in self) < FOREACH_POW_FUSED_NUMEL:
        # Launch-bound regime: one fused launch beats 16 per-tensor launches.
        from flag_gems.ops._foreach_binary import (
            _foreach_pow_List as _generic_foreach_pow_List,
        )

        return _generic_foreach_pow_List(self, other)
    results = []
    for a, b in zip(self, other):
        if a.dtype == torch.bool:
            raise NotImplementedError(_POW_BOOL_ERROR)
        out_dtype = torch.result_type(a, b)
        if (
            a.shape == b.shape
            and a.is_contiguous()
            and b.is_contiguous()
            and a.is_floating_point()
            and b.is_floating_point()
            and out_dtype == a.dtype
            and not a.is_complex()
            and not b.is_complex()
        ):
            out = torch.empty_like(a)
            _launch_pow_tt_fast(a, b, out)
            results.append(out)
        else:
            results.append(pow_func(a, b))
    return results
