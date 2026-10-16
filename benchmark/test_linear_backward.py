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


LINEAR_BACKWARD_SHAPES = [(64, 512), (128, 1024), (256, 2048), (512, 4096)]


class LinearBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LINEAR_BACKWARD_SHAPES
        self.shape_desc = "batch, in_features"


def _case_fn(shape, dtype):
    del dtype
    batch, in_features = shape
    out_features = in_features * 2
    yield base.BenchmarkCasePlan(
        shape={
            "input": [batch, in_features],
            "grad_output": [batch, out_features],
            "weight": [out_features, in_features],
        },
        params={"output_mask": [True, True, True]},
        builder_args=(batch, in_features, out_features),
    )


def _build_inputs_fn(plan, dtype, device):
    batch, in_features, out_features = plan.builder_args
    torch.manual_seed(0)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=device
    )
    return input_tensor, grad_output, weight, (True, True, True)


def _reference(input_tensor, grad_output, weight, output_mask):
    grad_input = grad_output @ weight if output_mask[0] else None
    grad_weight = grad_output.transpose(-2, -1) @ input_tensor if output_mask[1] else None
    grad_bias = grad_output.sum(dim=0) if output_mask[2] else None
    return grad_input, grad_weight, grad_bias


@pytest.mark.linear_backward
def test_linear_backward():
    LinearBackwardBenchmark(
        op_name="linear_backward",
        torch_op=_reference,
        gems_op=flag_gems.linear_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32, torch.float16],
    ).run()
