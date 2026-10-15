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

from . import base, consts, utils


def _case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"max": 3.14},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    return inp1, 3.14


@pytest.mark.clamp_max
def test_clamp_max():
    bench = base.GenericBenchmark(
        op_name="clamp_max",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.clamp_max,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.clamp_max_
def test_clamp_max_inplace():
    bench = base.GenericBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="clamp_max_",
        torch_op=torch.clamp_max_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
