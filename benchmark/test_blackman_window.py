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

import flag_gems

from . import base, consts


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"window_length": list(shape)},
        params={"periodic": True},
        builder_args=(shape, True),
    )

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={"window_length": list(shape)},
            params={"periodic": False},
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


@pytest.mark.blackman_window
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_blackman_window():
    bench = base.GenericBenchmark(
        op_name="blackman_window",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.blackman_window,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.blackman_window_periodic
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_blackman_window_periodic():
    bench = base.GenericBenchmark(
        op_name="blackman_window_periodic",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.blackman_window,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
