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

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(4, 16, 8), (8, 32, 16), (4, 64, 16), (8, 128, 32)]
QGROUP_SIZES = (2, 4, 8, 16)


def _reference_mm(a, mat2, q_group_size, q_scale, q_zeros):
    k = a.shape[1]
    packed = mat2.to(torch.int32)
    values = torch.empty((k, mat2.shape[0]), dtype=torch.float32, device=a.device)
    for index in range(k):
        byte = packed[:, index // 2]
        nibble = byte & 0xF if index % 2 == 0 else (byte >> 4) & 0xF
        group = index // q_group_size
        values[index] = (nibble.to(torch.float32) - q_zeros[group]) * q_scale[group]
    return torch.mm(a.to(torch.float32), values).to(a.dtype)


def _gems_op(a, mat2, q_group_size, q_scale, q_zeros):
    return flag_gems._weight_int4pack_mm_with_scales_and_zeros(
        a, mat2, q_group_size, q_scale, q_zeros
    )


class WeightInt4Benchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "M, K, N"


def _case_fn(shape, dtype):
    del dtype
    m, k, n = tuple(shape)
    for q_group_size in QGROUP_SIZES:
        if k % q_group_size == 0:
            yield base.BenchmarkCasePlan(
                shape={"self": [m, k], "mat2": [n, k // 2]},
                params={"qGroupSize": q_group_size},
                builder_args=(m, k, n, q_group_size),
            )


def _build_inputs_fn(plan, dtype, device):
    m, k, n, q_group_size = plan.builder_args
    torch.manual_seed(0)
    a = torch.randn((m, k), dtype=dtype, device=device) * 0.1
    raw = torch.randint(0, 16, (k, n), dtype=torch.int32, device=device)
    mat2 = ((raw[1::2] << 4) | raw[0::2]).t().contiguous().to(torch.uint8)
    groups = k // q_group_size
    q_scale = torch.randn((groups, n), dtype=dtype, device=device).abs() + 0.01
    q_zeros = torch.randn((groups, n), dtype=dtype, device=device) * 2.0
    return a, mat2, q_group_size, q_scale, q_zeros


@pytest.mark.weight_int4pack_mm_with_scales_and_zeros
def test_weight_int4pack_mm_with_scales_and_zeros():
    WeightInt4Benchmark(
        op_name="weight_int4pack_mm_with_scales_and_zeros",
        torch_op=_reference_mm,
        gems_op=_gems_op,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
