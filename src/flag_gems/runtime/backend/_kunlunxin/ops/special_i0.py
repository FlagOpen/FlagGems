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

# Kunlunxin (XPU) override of special_i0 / special_i0.out.
#
# aten.special_i0 is mathematically identical to aten.i0 (zeroth order modified
# Bessel function of the first kind). The generic KernelGen kernel uses a plain
# BLOCK_SIZE=1024 cdiv grid with no XPU launch tuning, which leaves this
# memory-bound pointwise op ~70x slower than it should be on XPU. This override
# reuses the tuned pointwise_dynamic Bessel-I0 kernel (i0_func) that auto-sizes
# the grid to the 12x64 cluster/core topology and streams coalesced DMA tiles.
import logging

from .i0 import i0_func

logger = logging.getLogger(__name__)


def special_i0(x):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I0")
    return i0_func(x)


def special_i0_out(x, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I0_OUT")
    return i0_func(x, out0=out)
