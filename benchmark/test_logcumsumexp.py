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


def _case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"dim": 1},
        builder_args=(shape,),
    )


def logcumsumexp_build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    return inp, 1


@pytest.mark.logcumsumexp
def test_logcumsumexp():
    bench = base.GenericBenchmark2DOnly(
        op_name="logcumsumexp",
        case_fn=_case_fn,
        build_inputs_fn=logcumsumexp_build_inputs_fn,
        torch_op=torch.logcumsumexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def logcumsumexp_out_build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    out = torch.empty_like(inp)
    return inp, 1, {"out": out}


@pytest.mark.logcumsumexp_out
def test_logcumsumexp_out():
    bench = base.GenericBenchmark2DOnly(
        op_name="logcumsumexp_out",
        case_fn=_case_fn,
        build_inputs_fn=logcumsumexp_out_build_inputs_fn,
        torch_op=torch.ops.aten.logcumsumexp.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
