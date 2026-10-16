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

import math

import pytest
import torch

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


_DEFAULT_SHAPES = [(1,), (16,), (256,), (4096,)]


class HammingWindowBenchmark(OperatorBenchmark):
    DEFAULT_SHAPE_DESC = "window_length"

    def set_shapes(self, shape_file_path):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SHAPES)


def _case_fn(shape, dtype):
    del dtype
    window_length = math.prod(shape)
    yield base.BenchmarkCasePlan(
        shape={"output": [window_length]},
        params={"periodic": True, "alpha": 0.54},
        builder_args=(window_length, True, 0.54),
    )
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={"output": [window_length]},
            params={"periodic": False, "alpha": 0.5},
            builder_args=(window_length, False, 0.5),
        )


def _build_inputs_fn(plan, dtype, device):
    window_length, periodic, alpha = plan.builder_args
    return (
        {
            "window_length": window_length,
            "periodic": periodic,
            "alpha": alpha,
            "dtype": dtype,
            "device": device,
        },
    )


@pytest.mark.hamming_window_periodic_alpha
def test_hamming_window_periodic_alpha():
    bench = HammingWindowBenchmark(
        op_name="hamming_window_periodic_alpha",
        torch_op=torch.hamming_window,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
