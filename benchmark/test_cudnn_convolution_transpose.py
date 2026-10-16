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

SHAPES = [(1, 8, 32, 32), (4, 32, 16, 16)]


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
    n, c, h, w = shape
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((c, c, 3, 3), dtype=dtype, device=device)
    return inp, weight, [1, 1], [0, 0], [1, 1], [1, 1], 1, False, False, False


@pytest.mark.cudnn_convolution_transpose
def test_cudnn_convolution_transpose():
    OperatorCaseBenchmark(
        op_name="cudnn_convolution_transpose",
        torch_op=torch.cudnn_convolution_transpose,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float16, torch.float32],
    ).run()
