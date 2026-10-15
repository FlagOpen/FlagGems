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


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"p": 0.5, "train": True},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    # alpha_dropout takes (input, p, train) — p defaults to 0.5
    return inp, 0.5, True


@pytest.mark.alpha_dropout
def test_alpha_dropout():
    bench = base.GenericBenchmarkExcluse1D(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="alpha_dropout",
        torch_op=torch.alpha_dropout,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
