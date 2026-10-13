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

import triton
import triton.language as tl

from flag_gems.utils import libentry


@libentry()
@triton.jit
def _nearest2d_backward_channels_last(
    GO,
    GI,
    N: tl.constexpr,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SCALE_H: tl.constexpr,
    SCALE_W: tl.constexpr,
    H_STEPS: tl.constexpr,
    W_STEPS: tl.constexpr,
    PIXELS: tl.constexpr,
    BLOCK: tl.constexpr,
    PATCH: tl.constexpr,
):
    # Each core owns consecutive tiles of input pixels. Read contiguous
    # channel runs into UB, then gather the short inverse windows locally.
    tiles: tl.constexpr = triton.cdiv(IW, PIXELS)
    tasks: tl.constexpr = N * IH * tiles
    count = tl.cdiv(tasks, tl.num_programs(0))
    first = tl.program_id(0) * count
    index = tl.arange(0, BLOCK)
    col = tl.floor(tl.div_rn(index.to(tl.float32), C))
    channel = (index.to(tl.float32) - col * C).to(tl.int32)
    patch_index = tl.arange(0, PATCH)
    for task in range(first, tl.minimum(first + count, tasks)):
        row = task // tiles
        batch = row // IH
        iy = row - batch * IH
        first_col = (task - row * tiles) * PIXELS
        ix = first_col + col
        # Keep a prefix mask so stores do not rewrite neighboring row tails.
        valid = index < tl.minimum(PIXELS, IW - first_col) * C
        h0 = tl.minimum(tl.ceil(iy.to(tl.float32) * SCALE_H), OH).to(tl.int32)
        h1 = tl.minimum(tl.ceil((iy + 1).to(tl.float32) * SCALE_H), OH).to(tl.int32)
        w0 = tl.minimum(tl.ceil(ix * SCALE_W), OW)
        w1 = tl.minimum(tl.ceil((ix + 1) * SCALE_W), OW)
        patch_col = tl.minimum(tl.ceil(first_col.to(tl.float32) * SCALE_W), OW).to(
            tl.int32
        )
        acc = tl.full((BLOCK,), 0, tl.float32)
        for dy in tl.static_range(H_STEPS):
            oy = h0 + dy
            # Clamp the source row and mask its contribution below. A scalar
            # FP32-derived height condition in the load mask fails to lower on
            # CANN 9.1.1; the contiguous column mask is sufficient for safety.
            values = tl.load(
                GO
                + ((batch * OH + tl.minimum(oy, OH - 1)) * OW + patch_col) * C
                + patch_index,
                patch_index < (OW - patch_col) * C,
                other=0,
            ).to(tl.float32)
            for dx in tl.static_range(W_STEPS):
                ox = w0 + dx
                local = (ox.to(tl.int32) - patch_col) * C + channel
                local = tl.minimum(tl.maximum(local, 0), PATCH - 1)
                if PATCH == 1:
                    value = tl.sum(values, 0)
                else:
                    value = tl.gather(values, local, 0)
                acc += tl.where(valid & (ox < w1) & (oy < h1), value, 0.0)
        tl.store(GI + row * IW * C + first_col * C + index, acc, valid)
