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


def _binary_case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape), "other": list(shape)},
        builder_args=(shape,),
    )


def _binary_build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = utils.generate_tensor_input(shape, cur_dtype, device)
    return inp1, inp2


@pytest.mark.special_xlogy
def test_special_xlogy():
    bench = base.GenericBenchmark(
        op_name="special_xlogy",
        case_fn=_binary_case_fn,
        build_inputs_fn=_binary_build_inputs_fn,
        torch_op=torch.special.xlogy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_xlogy_
def test_special_xlogy_():
    bench = base.GenericBenchmark(
        op_name="special_xlogy_",
        case_fn=_binary_case_fn,
        build_inputs_fn=_binary_build_inputs_fn,
        torch_op=lambda x, y: torch.ops.aten.xlogy_(x, y),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
