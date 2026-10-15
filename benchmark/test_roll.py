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

from . import base, consts, utils


def _roll_kwargs(shape):
    if len(shape) > 1:
        return {"shifts": (1, 2), "dims": (0, 1)}
    return {"shifts": 1, "dims": 0}


def _case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params=_roll_kwargs(shape),
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    return inp, _roll_kwargs(shape)


@pytest.mark.roll
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_roll():
    bench = base.GenericBenchmark(
        op_name="roll",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.roll,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
