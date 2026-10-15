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


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"window": list(shape)},
        params={"window_length": math.prod(shape), "periodic": True},
        builder_args=(shape, True),
    )

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={"window": list(shape)},
            params={"window_length": math.prod(shape), "periodic": False},
            builder_args=(shape, False),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, periodic = plan.builder_args
    return (
        {
            "window_length": math.prod(shape),
            "periodic": periodic,
            "dtype": dtype,
            "device": device,
        },
    )


@pytest.mark.hamming_window
def test_hamming_window():
    bench = base.GenericBenchmark(
        op_name="hamming_window",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.hamming_window,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.hamming_window_periodic
def test_hamming_window_periodic():
    bench = base.GenericBenchmark(
        op_name="hamming_window_periodic",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.hamming_window,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
