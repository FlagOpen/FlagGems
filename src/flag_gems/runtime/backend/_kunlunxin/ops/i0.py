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

# Kunlunxin (XPU) override of i0 / i0.out.
#
# The generic KernelGen i0 kernel uses a plain BLOCK_SIZE=1024 cdiv grid with
# no XPU launch tuning, so it leaves the memory-bound Bessel-I0 pointwise op
# slow. This override moves the op onto the vendor pointwise_dynamic framework
# with a tuned CodeGenConfig (kunlunAutoGrid + unroll), which auto-sizes the
# grid to the 12x64 cluster/core topology and streams in coalesced DMA tiles.
# The modified-Bessel-I0 polynomial (Abramowitz & Stegun 9.8.1/9.8.2) is
# evaluated in fp32 and cast back to the promoted output dtype.
import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def i0_func(x):
    xf = x.to(tl.float32)
    ax = tl.abs(xf)

    # Small region: |x| <= 3.75
    t = xf / 3.75
    y = t * t
    p_small = 1.0 + y * (
        3.5156229
        + y
        * (
            3.0899424
            + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))
        )
    )

    # Large region: |x| > 3.75
    yb = 3.75 / ax
    p_big = 0.39894228 + yb * (
        0.01328592
        + yb
        * (
            0.00225319
            + yb
            * (
                -0.00157565
                + yb
                * (
                    0.00916281
                    + yb
                    * (
                        -0.02057706
                        + yb * (0.02635537 + yb * (-0.01647633 + yb * 0.00392377))
                    )
                )
            )
        )
    )
    res_big = tl.exp(ax) * p_big / tl.sqrt(ax)

    return tl.where(ax <= 3.75, p_small, res_big)


def i0(A):
    logger.debug("GEMS_KUNLUNXIN I0")
    return i0_func(A)


def i0_out(A, out):
    logger.debug("GEMS_KUNLUNXIN I0_OUT")
    return i0_func(A, out0=out)
