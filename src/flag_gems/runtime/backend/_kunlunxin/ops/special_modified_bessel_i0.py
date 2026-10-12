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

# Kunlunxin (XPU) override of special_modified_bessel_i0 / .out.
#
# This is the same zeroth order modified Bessel function of the first kind as
# aten.i0 / aten.special_i0. The generic KernelGen kernel uses a plain
# BLOCK_SIZE=1024 cdiv grid with no XPU launch tuning. This override reuses the
# tuned pointwise_dynamic Bessel-I0 kernel (i0_func) that auto-sizes the grid to
# the 12x64 cluster/core topology and streams coalesced DMA tiles.
import logging

from .i0 import i0_func

logger = logging.getLogger(__name__)


def special_modified_bessel_i0(x):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_MODIFIED_BESSEL_I0")
    return i0_func(x)


def special_modified_bessel_i0_out(x, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_MODIFIED_BESSEL_I0_OUT")
    return i0_func(x, out0=out)
