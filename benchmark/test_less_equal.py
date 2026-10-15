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


def _scalar_case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"inp": list(shape)},
        builder_args=(shape,),
    )


def _scalar_build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    return inp, 0


@pytest.mark.less_equal
def test_less_equal():
    bench = base.BinaryPointwiseBenchmark(
        op_name="less_equal",
        torch_op=torch.less_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.less_equal_scalar
def test_less_equal_scalar():
    bench = base.GenericBenchmark(
        case_fn=_scalar_case_fn,
        build_inputs_fn=_scalar_build_inputs_fn,
        op_name="less_equal_scalar",
        torch_op=torch.less_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
