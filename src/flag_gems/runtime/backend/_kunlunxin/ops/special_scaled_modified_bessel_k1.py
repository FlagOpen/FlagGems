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

# Kunlunxin (XPU) override of special_scaled_modified_bessel_k1 / .out.
#
# The generic KernelGen kernel uses a plain BLOCK_SIZE=256 cdiv grid with no XPU
# launch tuning, leaving this memory-bound pointwise op ~100x slower than it
# should be on XPU (~200ms Gems latency). This override moves it onto the vendor
# pointwise_dynamic framework with a tuned CodeGenConfig (kunlunAutoGrid +
# unroll) that auto-sizes the grid to the 12x64 cluster/core topology and
# streams coalesced DMA tiles. The scaled-K1 math (three-region Cephes/Chebyshev
# + asymptotic series) is evaluated in fp32 and cast back to the output dtype.
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
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=4,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def k1_func(x):
    x_f32 = x.to(tl.float32)
    zero = 0.0
    ax = tl.abs(x_f32)
    is_zero = ax < 1e-30
    is_negative = x_f32 < zero

    # Small region: 0 < x <= 2.0 ; scaled_K1 = exp(x) * (log(x/2)*I1(x) + P(x)/x)
    y_i1 = x_f32 / 3.75
    y2_i1 = y_i1 * y_i1
    p_i1 = 0.00032411
    p_i1 = 0.00301532 + y2_i1 * p_i1
    p_i1 = 0.02658733 + y2_i1 * p_i1
    p_i1 = 0.15084934 + y2_i1 * p_i1
    p_i1 = 0.51498869 + y2_i1 * p_i1
    p_i1 = 0.87890594 + y2_i1 * p_i1
    p_i1 = 0.5 + y2_i1 * p_i1
    i1_val = x_f32 * p_i1

    y_cheb = x_f32 - 1.0
    b0 = 0.0
    b1 = 0.0
    b2 = b1
    b1 = b0
    b0 = -3.662620162661560e-09 + 2.0 * y_cheb * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -5.350617072358099e-08 + 2.0 * y_cheb * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -1.045185163440483e-06 + 2.0 * y_cheb * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -1.201112421320492e-05 + 2.0 * y_cheb * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -1.699042469646379e-04 + 2.0 * y_cheb * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -1.426485944971723e-03 + 2.0 * y_cheb * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -1.225928401715418e-02 + 2.0 * y_cheb * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -6.428271235627786e-02 + 2.0 * y_cheb * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -1.767199860308625e-01 + 2.0 * y_cheb * b1 - b2
    b2 = b1
    b1 = b0
    b0 = -2.944128552518444e-01 + 2.0 * y_cheb * b1 - b2
    b2 = b1
    b1 = b0
    b0 = 1.658032209918573e00 + 2.0 * y_cheb * b1 - b2
    cheb_val = (b0 - b2) / 2.0

    log_term = tl.log(tl.maximum(x_f32, 1e-30) / 2.0)
    k1_small = log_term * i1_val + cheb_val / tl.maximum(x_f32, 1e-30)
    small_result = tl.exp(x_f32) * k1_small

    # Mid region: 2.0 < x <= 8.0
    t_mid = 1.0 / ax
    poly_mid = 1.787118672412693
    poly_mid = 8.623190654362435 + t_mid * poly_mid
    poly_mid = -31.24007105377005 + t_mid * poly_mid
    poly_mid = 39.94627099526494 + t_mid * poly_mid
    poly_mid = -27.61325465617906 + t_mid * poly_mid
    poly_mid = 11.54410129937854 + t_mid * poly_mid
    poly_mid = -2.958011161687468 + t_mid * poly_mid
    poly_mid = 0.352116615179864 + t_mid * poly_mid
    poly_mid = 0.424654857840721 + t_mid * poly_mid
    poly_mid = 1.255087565900909 + t_mid * poly_mid
    mid_result = poly_mid / tl.sqrt(ax)

    # Large region: x > 8.0
    t_large = 1.0 / ax
    poly_large = 1.0 + t_large * (
        0.375
        + t_large
        * (
            -0.1171875
            + t_large
            * (
                0.1025390625
                + t_large
                * (-0.144195556640625 + t_large * 0.277576446533203125)
            )
        )
    )
    big_result = 1.2533141373155001 * poly_large / tl.sqrt(ax)

    use_small = ax <= 2.0
    use_mid = (ax > 2.0) & (ax <= 8.0)
    result = tl.where(
        use_small, small_result, tl.where(use_mid, mid_result, big_result)
    )
    result = tl.where(is_zero, float("inf"), result)
    result = tl.where(is_negative, float("nan"), result)
    return result


def special_scaled_modified_bessel_k1(x):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SCALED_MODIFIED_BESSEL_K1")
    return k1_func(x)


def special_scaled_modified_bessel_k1_out(x, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SCALED_MODIFIED_BESSEL_K1_OUT")
    return k1_func(x, out0=out)
