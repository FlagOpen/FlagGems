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


def _inplace_case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"p": 0.5},
        builder_args=(shape,),
    )


def _inplace_build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    self = torch.randn(shape, dtype=cur_dtype, device=device)
    p = 0.5
    return self, p


@pytest.mark.bernoulli_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_bernoulli_inplace():
    bench = base.GenericBenchmark(
        op_name="bernoulli_",
        case_fn=_inplace_case_fn,
        build_inputs_fn=_inplace_build_inputs_fn,
        torch_op=torch.Tensor.bernoulli_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    return (torch.rand(shape, dtype=cur_dtype, device=device),)


@pytest.mark.bernoulli
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_bernoulli():
    bench = base.GenericBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="bernoulli",
        torch_op=torch.bernoulli,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
