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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


_CASES = [
    ((1024,), (1024,)),
    ((1024,), (1024, 512)),
    ((512, 1024), (1024,)),
    ((512, 1024), (1024, 256)),
    ((8, 512, 1024), (8, 1024, 256)),
    ((1, 512, 1024), (8, 1024, 256)),
]


def _case_fn(shape, dtype):
    del shape, dtype
    for left_shape, right_shape in _CASES:
        yield base.BenchmarkCasePlan(
            shape={"self": list(left_shape), "other": list(right_shape)},
            params={"kind": f"{len(left_shape)}d_x_{len(right_shape)}d"},
            builder_args=(left_shape, right_shape),
        )


def _build_inputs_fn(plan, dtype, device):
    left_shape, right_shape = plan.builder_args
    torch.manual_seed(0)
    left = torch.randn(left_shape, dtype=dtype, device=device)
    right = torch.randn(right_shape, dtype=dtype, device=device)
    return left, right


@pytest.mark.linalg_matmul
def test_linalg_matmul():
    bench = OperatorBenchmark(
        op_name="linalg_matmul",
        torch_op=torch.ops.aten.linalg_matmul.default,
        gems_op=flag_gems.linalg_matmul,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
