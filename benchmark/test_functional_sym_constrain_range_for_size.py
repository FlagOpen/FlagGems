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
        shape={"dep_token": list(shape)},
        params={"size": 5, "min": 1, "max": 10},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    dep_token = utils.generate_tensor_input(shape, cur_dtype, device)
    return 5, 1, 10, dep_token


@pytest.mark.functional_sym_constrain_range_for_size
def test_functional_sym_constrain_range_for_size():
    bench = base.GenericBenchmark(
        op_name="functional_sym_constrain_range_for_size",
        torch_op=torch.ops.aten._functional_sym_constrain_range_for_size,
        dtypes=consts.FLOAT_DTYPES,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
    )
    bench.run()
