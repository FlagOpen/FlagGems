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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops import scalar_tensor as _gems_scalar_tensor

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems.ops.__rshift__")

# Two codegen variants for the two latency regimes on XPU (mirrors the validated
# sibling override _kunlunxin/ops/bitwise_right_shift.py):
# - small tensors (numel <= 4096): launch-bound -> 1-CTA (kunlunAutoGrid) path
#   keeps the whole tensor in one CTA.
# - medium/large tensors: grid-stride path with explicit unroll 16, the sweet
#   spot measured for this op family on XPU.
config_small = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)
config_large = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_small)
@triton.jit
def rshift_kernel_small(a, b):
    return a >> b


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_large)
@triton.jit
def rshift_kernel_large(a, b):
    return a >> b


def __rshift__(self, other, *, out=None):
    """``__rshift__`` for Kunlunxin (tensor/scalar and ``.out`` overloads).

    The generic ``_rshift_tensor_kernel``/``_rshift_scalar_kernel`` run without
    an XPU CodeGenConfig, so every shape recompiles and the scalar variant keeps
    a runtime scalar operand that blocks vectorization.  Here the scalar is
    materialized as a device tensor (so the kernel is the same tensor-tensor
    fast path) and the two size regimes use the validated small/large recipes.
    """
    logger.debug("GEMS_KUNLUNXIN __RSHIFT__")

    if not torch.is_tensor(other):
        other = _gems_scalar_tensor(other, dtype=self.dtype, device=self.device)

    large = self.numel() > 4096
    if out is not None:
        if large:
            rshift_kernel_large(self, other, out0=out)
        else:
            rshift_kernel_small(self, other, out0=out)
        return out
    if large:
        return rshift_kernel_large(self, other)
    return rshift_kernel_small(self, other)
