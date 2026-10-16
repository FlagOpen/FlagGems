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
    (384, 384, 384),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (128, 256, 512),
    (256, 1024, 4096),
]


class MatmulBiasActivationBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_CASES)


def _case_fn(shape, dtype):
    del dtype
    m, n, k = shape
    yield base.BenchmarkCasePlan(
        shape={"input": [m, k], "weight": [k, n], "bias": [n]},
        params={"M": m, "N": n, "K": k, "activation": "relu"},
        builder_args=(m, n, k),
    )


def _build_inputs_fn(plan, dtype, device):
    m, n, k = plan.builder_args
    torch.manual_seed(0)
    input_tensor = torch.randn((m, k), dtype=dtype, device=device)
    weight = torch.randn((k, n), dtype=dtype, device=device)
    bias = torch.randn((n,), dtype=dtype, device=device)
    return input_tensor, weight, bias


def _reference(input_tensor, weight, bias):
    return torch.relu(torch.mm(input_tensor, weight) + bias)


@pytest.mark.matmul_bias_activation
def test_matmul_bias_activation():
    bench = MatmulBiasActivationBenchmark(
        op_name="matmul_bias_activation",
        torch_op=_reference,
        gems_op=flag_gems.matmul_bias_activation,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
