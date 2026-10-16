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

import math

import pytest
import torch

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(16, 16, 64), (16, 8, 128, 128)]


class OperatorCaseBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "operator-specific shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(channels, dtype=dtype, device=device)
    bias = torch.randn(channels, dtype=dtype, device=device)
    running_mean = torch.randn(channels, dtype=dtype, device=device)
    running_var = torch.rand(channels, dtype=dtype, device=device) + 0.1
    return inp, weight, bias, running_mean, running_var, 0.1, 1e-5


@pytest.mark.batch_norm_with_update
def test_batch_norm_with_update():
    OperatorCaseBenchmark(
        op_name="batch_norm_with_update",
        torch_op=torch.ops.aten._batch_norm_with_update,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        fresh_inputs=True,
    ).run()
