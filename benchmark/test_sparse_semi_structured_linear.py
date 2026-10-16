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

LINEAR_SHAPES = [(16, 32), (64, 128), (256, 512), (128, 1024), (512, 2048)]


def _case_fn(shape, dtype):
    del dtype
    m, k = shape
    for use_bias in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"input": [m, k], "weight": [k, k], "meta": [k, k // 4]},
            params={"bias": use_bias, "sparsity": "2:4"},
            builder_args=(m, k, use_bias),
        )


def _build_inputs_fn(plan, dtype, device):
    m, k, use_bias = plan.builder_args
    torch.manual_seed(0)
    input_tensor = torch.randn(m, k, dtype=dtype, device=device)
    base_weight = torch.randn(k, k, dtype=dtype, device=device)
    choice = torch.randint(0, 2, (k, k // 4), device=device).bool()
    grouped = base_weight.view(k, k // 4, 4)
    keep = torch.stack((choice, choice, ~choice, ~choice), dim=-1)
    weight = torch.where(keep, grouped, torch.zeros_like(grouped)).reshape(k, k)
    meta = choice.to(torch.int8)
    if use_bias:
        return input_tensor, weight, meta, torch.randn(k, dtype=dtype, device=device)
    return input_tensor, weight, meta


def _torch_ref(input_tensor, weight, meta, bias=None):
    grouped = weight.view(weight.shape[0], weight.shape[1] // 4, 4)
    choice = meta.to(torch.bool)
    keep = torch.stack((choice, choice, ~choice, ~choice), dim=-1)
    masked = torch.where(keep, grouped, torch.zeros_like(grouped)).reshape_as(weight)
    output = input_tensor @ masked.t()
    return output if bias is None else output + bias


class SparseLinearBenchmark(OperatorBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = LINEAR_SHAPES
        self.shape_desc = "M, K"

    def get_tflops(self, op, *args, **kwargs):
        m, k = args[0].shape
        return m * k * k


@pytest.mark.sparse_semi_structured_linear
def test_sparse_semi_structured_linear():
    bench = SparseLinearBenchmark(
        op_name="sparse_semi_structured_linear",
        torch_op=_torch_ref,
        gems_op=flag_gems._sparse_semi_structured_linear,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
