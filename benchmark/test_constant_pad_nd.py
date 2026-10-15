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

import random

import pytest
import torch

import flag_gems

from . import base, consts


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"value": 1.5},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, device=device, dtype=dtype)
    rank = inp.ndim
    pad = [random.randint(0, 10) for _ in range(rank * 2)]
    value = 1.5
    return inp, {"pad": pad, "value": value}


@pytest.mark.constant_pad_nd
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_constant_pad_nd():
    bench = base.GenericBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="constant_pad_nd",
        torch_op=torch.constant_pad_nd,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
