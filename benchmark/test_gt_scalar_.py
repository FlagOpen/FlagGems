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

from . import base, consts, utils


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"other": 0.5},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, 0.5


@pytest.mark.gt_scalar_
def test_gt_scalar_():
    bench = base.GenericBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="gt_scalar_",
        torch_op=lambda a, b: a.gt_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
