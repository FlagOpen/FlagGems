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


LDL_SOLVE_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (32, 7),
]
LDL_SOLVE_DTYPES = [torch.float32, torch.float64, torch.complex64]
if flag_gems.runtime.device.support_fp64:
    LDL_SOLVE_DTYPES.append(torch.complex128)


class LinalgLdlSolveBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LDL_SOLVE_SHAPES
        self.shape_desc = "N, K"


def _case_fn(shape, dtype):
    del dtype
    n, k = shape
    yield base.BenchmarkCasePlan(
        shape={"LD": [n, n], "pivots": [n], "B": [n, k]},
        params={"hermitian": False},
        builder_args=(n, k),
    )


def _build_inputs_fn(plan, dtype, device):
    n, k = plan.builder_args
    torch.manual_seed(0)
    matrix = torch.randn((n, n), dtype=dtype, device=device)
    matrix = matrix @ matrix.mT + n * torch.eye(n, dtype=dtype, device=device)
    ld, pivots, info = torch.linalg.ldl_factor_ex(matrix)
    assert info.eq(0).all()
    rhs = torch.randn((n, k), dtype=dtype, device=device)
    return ld, pivots, rhs, {"hermitian": False}


@pytest.mark.linalg_ldl_solve
def test_linalg_ldl_solve():
    LinalgLdlSolveBenchmark(
        op_name="linalg_ldl_solve",
        torch_op=torch.linalg.ldl_solve,
        gems_op=flag_gems.linalg_ldl_solve,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=LDL_SOLVE_DTYPES,
    ).run()
