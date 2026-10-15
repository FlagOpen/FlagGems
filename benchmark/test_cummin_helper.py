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


def _cummin_helper_case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def _cummin_helper_build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64, device=device)
    dim = 1 if len(shape) >= 2 else 0
    return inp, values, indices, dim


@pytest.mark.cummin_helper
def test_cummin_helper():
    bench = base.GenericBenchmark2DOnly(
        op_name="cummin_helper",
        case_fn=_cummin_helper_case_fn,
        build_inputs_fn=_cummin_helper_build_inputs_fn,
        torch_op=torch.ops.aten._cummin_helper,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
