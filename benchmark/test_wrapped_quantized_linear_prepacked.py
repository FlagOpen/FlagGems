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
from flag_gems.ops._wrapped_linear_prepack import unpack_linear_weight

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(2, 4, 8), (16, 64, 128), (64, 256, 256), (128, 512, 1024)]


def _torch_reference(
    inp,
    input_scale,
    input_zero_point,
    packed_weight,
    output_scale,
    output_zero_point,
    out_channel,
):
    weight, metadata, bias = unpack_linear_weight(packed_weight, out_channel)
    quantized_input = torch.clamp(
        torch.round(inp / input_scale) + input_zero_point, 0, 255
    )
    real_output = torch.matmul(
        quantized_input - input_zero_point,
        (weight.to(torch.float32) - metadata[1]).T,
    )
    real_output = real_output * input_scale * metadata[0] + bias
    quantized_output = torch.clamp(
        torch.round(real_output / output_scale) + output_zero_point, 0, 255
    )
    return (quantized_output - output_zero_point) * output_scale


class WrappedQuantizedLinearPrepackedBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "BATCH, OUT_FEATURES, IN_FEATURES"


def _case_fn(shape, dtype):
    del dtype
    batch, out_features, in_features = shape
    yield base.BenchmarkCasePlan(
        shape={
            "input": [batch, in_features],
            "weight": [out_features, in_features],
        },
        params={"out_channel": out_features},
        builder_args=(batch, out_features, in_features),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype
    batch, out_features, in_features = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn((batch, in_features), dtype=torch.float32, device=device)
    weight = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=device
    )
    bias = torch.randn(out_features, dtype=torch.float32, device=device)
    input_scale = torch.tensor(0.05, dtype=torch.float32, device=device)
    input_zero_point = torch.tensor(127, device=device)
    weight_scale = torch.tensor(0.03125, dtype=torch.float32, device=device)
    weight_zero_point = torch.tensor(-3, device=device)
    output_scale = torch.tensor(0.08, dtype=torch.float32, device=device)
    output_zero_point = torch.tensor(121, device=device)
    packed_weight = flag_gems._wrapped_linear_prepack(
        weight, weight_scale, weight_zero_point, bias
    )
    return (
        inp,
        input_scale,
        input_zero_point,
        packed_weight,
        output_scale,
        output_zero_point,
        out_features,
    )


@pytest.mark.wrapped_quantized_linear_prepacked
def test_wrapped_quantized_linear_prepacked_benchmark():
    WrappedQuantizedLinearPrepackedBenchmark(
        op_name="wrapped_quantized_linear_prepacked",
        torch_op=_torch_reference,
        gems_op=flag_gems._wrapped_quantized_linear_prepacked,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
