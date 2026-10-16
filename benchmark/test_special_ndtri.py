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

from . import base
from .generated_operator_utils import OperatorBenchmark


_DTYPES = [torch.float32, torch.float64]


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims)},
        params={"domain": [0.01, 0.99]},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.empty(shape, dtype=dtype, device=device).uniform_(0.01, 0.99)
    return (inp,)


@pytest.mark.special_ndtri
def test_special_ndtri():
    bench = OperatorBenchmark(
        op_name="special_ndtri",
        torch_op=torch.ops.aten.special_ndtri,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=_DTYPES,
    )
    bench.run()
