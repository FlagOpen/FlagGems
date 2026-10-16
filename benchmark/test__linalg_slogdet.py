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

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]


class OperatorCaseBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "(*B), M, N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp = inp + torch.eye(shape[-1], dtype=dtype, device=device) * shape[-1]
    return (inp,)


@pytest.mark.underscore_linalg_slogdet
@pytest.mark.linalg_slogdet
def test__linalg_slogdet():
    OperatorCaseBenchmark(
        op_name="_linalg_slogdet",
        torch_op=torch.ops.aten._linalg_slogdet.default,
        gems_op=flag_gems._linalg_slogdet,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
