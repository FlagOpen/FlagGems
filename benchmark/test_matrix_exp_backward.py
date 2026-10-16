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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

SQUARE_SHAPES = [
    (64, 64),
    (256, 256),
    (512, 512),
    (16, 64, 64),
    (16, 128, 128),
]


class MatrixExpBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SQUARE_SHAPES)
        self.shape_desc = "(..., N, N)"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "grad": list(dims)},
        params={},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    self_inp = torch.randn(shape, dtype=dtype, device=device)
    grad_inp = torch.randn(shape, dtype=dtype, device=device)
    return self_inp, grad_inp


@pytest.mark.matrix_exp_backward
def test_matrix_exp_backward():
    MatrixExpBackwardBenchmark(
        op_name="matrix_exp_backward",
        torch_op=torch.ops.aten.matrix_exp_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
