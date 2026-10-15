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


def _xlogy_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape), "other": list(shape)},
        builder_args=(shape,),
    )


def _xlogy_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    # keep ``other`` positive so ``log`` stays finite
    inp2 = torch.rand(shape, dtype=dtype, device=device) * 5.0 + 0.01
    return inp1, inp2


@pytest.mark.xlogy_
def test_xlogy_():
    bench = base.GenericBenchmark(
        op_name="xlogy_",
        torch_op=lambda a, b: a.xlogy_(b),
        case_fn=_xlogy_case_fn,
        build_inputs_fn=_xlogy_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _xlogy_scalar_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"other": 3.5},
        builder_args=(shape,),
    )


def _xlogy_scalar_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    return inp, 3.5


@pytest.mark.xlogy_scalar_other_
def test_xlogy_scalar_other_():
    bench = base.GenericBenchmark(
        op_name="xlogy_scalar_other_",
        torch_op=lambda a, b: a.xlogy_(b),
        case_fn=_xlogy_scalar_case_fn,
        build_inputs_fn=_xlogy_scalar_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
