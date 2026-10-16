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
    (384, 384, 384, "vector"),
    (1024, 1024, 1024, "vector"),
    (2048, 2048, 2048, "vector"),
    (128, 256, 512, "matrix"),
    (256, 1024, 4096, "vector"),
]


class MatmuladdBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_CASES)


def _case_fn(shape, dtype):
    del dtype
    m, n, k, bias_kind = shape
    bias_shape = [n] if bias_kind == "vector" else [m, n]
    yield base.BenchmarkCasePlan(
        shape={"input": [m, k], "other": [k, n], "bias": bias_shape},
        params={"M": m, "N": n, "K": k, "bias_kind": bias_kind},
        builder_args=(m, n, k, bias_kind),
    )


def _build_inputs_fn(plan, dtype, device):
    m, n, k, bias_kind = plan.builder_args
    torch.manual_seed(0)
    input_tensor = torch.randn((m, k), dtype=dtype, device=device)
    other = torch.randn((k, n), dtype=dtype, device=device)
    bias_shape = (n,) if bias_kind == "vector" else (m, n)
    bias = torch.randn(bias_shape, dtype=dtype, device=device)
    return input_tensor, other, bias


def _reference(input_tensor, other, bias):
    return torch.matmul(input_tensor, other) + bias


@pytest.mark.matmuladd
def test_matmuladd():
    bench = MatmuladdBenchmark(
        op_name="matmuladd",
        torch_op=_reference,
        gems_op=flag_gems.matmuladd,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
