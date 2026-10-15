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


@pytest.mark.logaddexp2
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logaddexp2():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logaddexp2",
        torch_op=torch.logaddexp2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def logaddexp2_out_case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"inp1": list(shape), "inp2": list(shape)},
        builder_args=(shape,),
    )


def logaddexp2_out_build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp1 = torch.randn(shape, dtype=cur_dtype, device=device)
    inp2 = torch.randn(shape, dtype=cur_dtype, device=device)
    out = torch.empty(shape, dtype=cur_dtype, device=device)
    return inp1, inp2, {"out": out}


@pytest.mark.logaddexp2_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logaddexp2_out():
    bench = base.GenericBenchmark(
        op_name="logaddexp2_out",
        torch_op=torch.logaddexp2,
        case_fn=logaddexp2_out_case_fn,
        build_inputs_fn=logaddexp2_out_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
