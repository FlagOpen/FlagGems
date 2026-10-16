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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


def _empty_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"output": list(dims)},
        params={"memory_format": "contiguous_format"},
        builder_args=(dims,),
    )


def _empty_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return shape, {
        "dtype": dtype,
        "device": device,
        "memory_format": torch.contiguous_format,
    }


def _empty_permuted_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    physical_layout = tuple(reversed(range(len(dims))))
    yield base.BenchmarkCasePlan(
        shape={"output": list(dims)},
        params={"physical_layout": list(physical_layout)},
        builder_args=(dims, physical_layout),
    )


def _empty_permuted_build_inputs_fn(plan, dtype, device):
    shape, physical_layout = plan.builder_args
    return shape, list(physical_layout), {"dtype": dtype, "device": device}


@pytest.mark.empty
def test_empty():
    bench = OperatorBenchmark(
        op_name="empty",
        torch_op=torch.empty,
        case_fn=_empty_case_fn,
        build_inputs_fn=_empty_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.empty_permuted
def test_empty_permuted():
    bench = OperatorBenchmark(
        op_name="empty_permuted",
        torch_op=torch.empty_permuted,
        case_fn=_empty_permuted_case_fn,
        build_inputs_fn=_empty_permuted_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
