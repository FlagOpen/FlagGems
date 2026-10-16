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

SHAPES = [
    (64, 64, 16, 16),
    (128, 128, 32, 32),
    (256, 256, 32, 32),
    (512, 512, 32, 32),
    (1024, 512, 64, 16),
    (1024, 1024, 64, 16),
]


class TensordotBenchmark(OperatorBenchmark):
    DEFAULT_SHAPE_DESC = "M, N, K1, K2"

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC


def _case_fn(shape, dtype):
    del dtype
    m, n, k1, k2 = shape
    yield base.BenchmarkCasePlan(
        shape={"self": [m, k1, k2], "other": [k1, k2, n]},
        params={"dims_self": [1, 2], "dims_other": [0, 1]},
        builder_args=(m, n, k1, k2),
    )


def _build_operands(m, n, k1, k2, dtype, device):
    torch.manual_seed(0)
    a = torch.randn((m, k1, k2), dtype=dtype, device=device)
    b = torch.randn((k1, k2, n), dtype=dtype, device=device)
    return a, b


def _build_inputs_fn(plan, dtype, device):
    m, n, k1, k2 = plan.builder_args
    a, b = _build_operands(m, n, k1, k2, dtype, device)
    out = torch.empty((m, n), dtype=dtype, device=device)
    return a, b, [1, 2], [0, 1], {"out": out}


@pytest.mark.tensordot_out
def test_tensordot_out():
    TensordotBenchmark(
        op_name="tensordot_out",
        torch_op=torch.ops.aten.tensordot.out,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
