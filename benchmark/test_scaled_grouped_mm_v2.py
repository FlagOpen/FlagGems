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


SCALED_GROUPED_MM_SHAPES = [
    (4, 64, 128, 128),
    (8, 128, 256, 256),
    (16, 256, 512, 512),
]
SCALED_GROUPED_MM_MORE_SHAPES = [
    (16, 512, 1024, 1024),
    (32, 256, 2048, 1024),
]


def _cuda_fp8_available():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


def _native_benchmark_available():
    return _cuda_fp8_available() and hasattr(torch, "_scaled_grouped_mm_v2")


class ScaledGroupedMMV2Benchmark(OperatorBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = SCALED_GROUPED_MM_SHAPES
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes = self.shapes + SCALED_GROUPED_MM_MORE_SHAPES
        self.shape_desc = "groups, M_per_group, N, K"

    def get_tflops(self, op, *args, **kwargs):
        del op, kwargs
        mat_a, mat_b = args[0], args[1]
        return mat_a.shape[0] * mat_a.shape[1] * mat_b.shape[-1] * 2


def _case_fn(shape, dtype):
    del dtype
    groups, m_per_group, n_per_group, k_size = shape
    total_m = groups * m_per_group
    yield base.BenchmarkCasePlan(
        shape={
            "mat_a": [total_m, k_size],
            "mat_b": [groups, k_size, n_per_group],
            "scale_a": [total_m],
            "scale_b": [groups, n_per_group],
            "offs": [groups],
        },
        params={
            "recipe_a": [1],
            "swizzle_a": [0],
            "recipe_b": [1],
            "swizzle_b": [0],
            "out_dtype": "bfloat16",
            "use_fast_accum": False,
        },
        builder_args=(groups, m_per_group, n_per_group, k_size),
    )


def _build_inputs_fn(plan, dtype, device):
    groups, m_per_group, n_per_group, k_size = plan.builder_args
    total_m = groups * m_per_group
    torch.manual_seed(0)
    mat_a = (torch.randn((total_m, k_size), device=device) * 0.25).to(dtype)
    mat_b = (
        (torch.randn((groups, n_per_group, k_size), device=device) * 0.25)
        .to(dtype)
        .transpose(-2, -1)
    )
    offs = (
        torch.arange(1, groups + 1, dtype=torch.int32, device=device) * m_per_group
    )
    scale_a = torch.linspace(0.75, 1.25, total_m, device=device)
    scale_b = torch.linspace(
        1.25, 0.75, groups * n_per_group, device=device
    ).reshape(groups, n_per_group)
    return (
        mat_a,
        mat_b,
        [scale_a],
        [1],
        [0],
        [scale_b],
        [1],
        [0],
        offs,
        None,
        torch.bfloat16,
        [],
        False,
    )


@pytest.mark.scaled_grouped_mm_v2
@pytest.mark.skipif(
    not _native_benchmark_available(),
    reason="native torch._scaled_grouped_mm_v2 benchmark requires CUDA FP8 on SM89+.",
)
def test_scaled_grouped_mm_v2():
    ScaledGroupedMMV2Benchmark(
        op_name="scaled_grouped_mm_v2",
        torch_op=torch._scaled_grouped_mm_v2,
        gems_op=flag_gems._scaled_grouped_mm_v2,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float8_e4m3fn],
    ).run()
