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

from flag_gems.ops.randn import pair_uniform_to_normal_fast
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)
from flag_gems.utils.shape_utils import broadcast_shapes, volume

logger = logging.getLogger(__name__)
UNROLL = 4


@pointwise_dynamic(
    is_tensor=[True, True, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_float(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_float_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_float_float(val, std, mean):
    return val * std + mean


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def normal_kernel_subblock(
    out_ptr,
    N,
    philox_seed,
    philox_offset,
    MAIN_BLOCK: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    # MAIN_BLOCK counts output elements; each Philox lane produces four values.
    num_sub_blocks: tl.constexpr = MAIN_BLOCK // (4 * SUB_BLOCK)
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, SUB_BLOCK).to(tl.int64)
    seed = philox_seed.to(tl.int64)
    offset = philox_offset.to(tl.int64)
    # A regular loop keeps the live vector width fixed at SUB_BLOCK.
    for sub_idx in range(num_sub_blocks):
        logical_sub = pid * num_sub_blocks + sub_idx
        counter = offset + logical_sub * SUB_BLOCK + lane
        c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
        zero = tl.full((SUB_BLOCK,), 0, tl.uint32)
        r0, r1, r2, r3 = tl.philox(seed, c0, c1, zero, zero)
        r0 = uint_to_uniform_float(r0)
        r1 = uint_to_uniform_float(r1)
        r2 = uint_to_uniform_float(r2)
        r3 = uint_to_uniform_float(r3)
        n0, n1 = pair_uniform_to_normal_fast(r0, r1)
        n2, n3 = pair_uniform_to_normal_fast(r2, r3)
        off0 = logical_sub * (4 * SUB_BLOCK) + lane
        off1 = off0 + SUB_BLOCK
        off2 = off1 + SUB_BLOCK
        off3 = off2 + SUB_BLOCK
        tl.store(out_ptr + off0, n0, mask=off0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off1, n1, mask=off1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off2, n2, mask=off2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off3, n3, mask=off3 < N, eviction_policy="evict_first")


def normal_distribution(shape, device, *, generator=None, out=None):
    if out is None:
        out = torch.empty(shape, device=device, dtype=torch.float32)
    N = volume(shape)
    if N == 0:
        return out
    sub_block = 1024
    main_block = max(
        sub_block * UNROLL, triton.next_power_of_2(triton.cdiv(N, 65535))
    )
    grid = triton.cdiv(N, main_block)
    # Reserve every sub-block, including masked lanes in the last main block.
    increment = grid * (main_block // UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(device):
        normal_kernel_subblock[(grid,)](
            out, N, philox_seed, philox_offset,
            MAIN_BLOCK=main_block, SUB_BLOCK=sub_block,
        )
    return out


def normal_tensor_tensor(mean, std, *, generator=None):
    logger.debug("GEMS NORMAL_TENSOR_TENSOR")
    shape = broadcast_shapes([mean.shape, std.shape])
    device = mean.device
    out = normal_distribution(shape, device)
    return transform_func_tensor_tensor(out, std, mean)


def normal_tensor_float(mean, std, *, generator=None):
    logger.debug("GEMS NORMAL_TENSOR_FLOAT")
    shape = mean.shape
    device = mean.device
    out = normal_distribution(shape, device)
    return transform_func_tensor_float(out, std, mean)


def normal_float_tensor(mean, std, *, generator=None):
    logger.debug("GEMS NORMAL_FLOAT_TENSOR")
    shape = std.shape
    device = std.device
    out = normal_distribution(shape, device)
    return transform_func_float_tensor(out, std, mean)


def normal_(self, mean=0, std=1, *, generator=None):
    logger.debug("GEMS NORMAL_")
    shape = self.shape
    device = self.device
    self = normal_distribution(shape, device, generator=generator, out=self)
    transform_func_float_float(self, std, mean, out0=self)
    return self
