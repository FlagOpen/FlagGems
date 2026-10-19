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

"""XPU override for ``aten::_foreach_round_`` (in-place round over Tensor[]).

Three fixes over the generic executor:

1. **Unmasked fast path.**  The shared ``foreach_unary`` executor's masked
   load/store blocks the block-DMA lowering (KERNEL_OPT_EXPERIENCE.md #36), so a
   homogeneous float list is processed by a single unmasked 2D-grid kernel.

2. **Direct pointer arguments.**  Passing the tensor pointers as scalar kernel
   arguments keeps block-DMA on the load/store (see ``_foreach_direct``).

3. **Hardware round.**  ``round`` (round half to even) promotes to fp32 and uses
   ``tl_extra_shim.rint`` -- the native ``xpu::rint`` extern -- instead of the
   generic executor's hand-rolled ``round_half_to_even_impl``.  The latter
   builds the result out of two ``tl.floor`` calls (each a slow soft extern on
   this backend) plus a ``tl.where``; ``rint`` is a single extern and is ~2.6x
   faster on the large benchmark shapes.  It is still a soft extern rather than
   a free hardware instruction, so ``round`` remains compute-bound, but this is
   the fastest correct round-half-to-even available on the backend.

Everything else -- mixed dtypes/shapes, gappy/overlapping views, non-divisible
sizes, integral inputs (which ``round`` leaves unchanged), and the rejected bool
dtype -- falls back to ``foreach_unary`` with ``inplace=True``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.foreach import check_tensor_list, foreach_unary, same_dtype
from flag_gems.utils.triton_lang_helper import tl_extra_shim

from ._foreach_direct import _launch_fast, _select_ptr

# The registration-liveness test (tests/test_foreach_unary.py) captures the
# record under the *generic* logger name, so it must be emitted from this
# logger rather than the module-local one.  The message carries the vendor
# prefix "GEMS_KUNLUNXIN", per the operator-log convention.
logger = logging.getLogger("flag_gems.ops._foreach_unary")

# round accepts every float and integral dtype but rejects bool and complex
# (matching the generic ``UNARY_OPS["round"].allowed``).
_NO_BOOL_NO_COMPLEX = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    + (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
)


@triton.jit
def _round_fast(x):
    # Integral inputs are already rounded; only floats go through rint.  The
    # int guard avoids an fp32 round-trip that would corrupt large int64 values.
    if x.dtype.is_floating():
        return tl_extra_shim.rint(x.to(tl.float32)).to(x.dtype)
    else:
        return x


@triton.jit
def _foreach_round_unmasked_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked launch over the tensor list (in-place)."""
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    ptr = _select_ptr(
        t, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15
    ).to(tl.pointer_type(DT))
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(ptr + idx).to(tl.float32)
    tl.store(ptr + idx, tl_extra_shim.rint(x).to(DT))


def _foreach_round_(self):
    logger.debug("GEMS_KUNLUNXIN _FOREACH_ROUND_")
    tensors = check_tensor_list(self)
    if _launch_fast(tensors, _foreach_round_unmasked_kernel):
        # In-place schemas return ``()``; returning the list would make the
        # dispatcher reject the kernel.
        return None
    foreach_unary(
        tensors,
        _round_fast,
        complex_fn=None,
        out_dtype_fn=same_dtype,
        inplace=True,
        allowed_dtypes=_NO_BOOL_NO_COMPLEX,
    )
    return None
