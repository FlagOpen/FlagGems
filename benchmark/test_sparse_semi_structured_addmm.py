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

ADDMM_SHAPES = [(64, 64), (128, 128), (256, 128), (512, 512)]


def _case_fn(shape, dtype):
    del dtype
    m, n = shape
    for alpha, beta in ((1.0, 1.0), (2.5, 0.5)):
        yield base.BenchmarkCasePlan(
            shape={"input": [m, n], "mat1": [m, 128], "meta": [m, 32], "mat2": [128, n]},
            params={"alpha": alpha, "beta": beta},
            builder_args=(m, n, alpha, beta),
        )


def _build_inputs_fn(plan, dtype, device):
    m, n, alpha, beta = plan.builder_args
    torch.manual_seed(0)
    return (
        torch.randn(m, n, dtype=dtype, device=device),
        torch.randn(m, 128, dtype=dtype, device=device),
        torch.randint(0, 2, (m, 32), dtype=torch.bool, device=device),
        torch.randn(128, n, dtype=dtype, device=device),
        {"alpha": alpha, "beta": beta},
    )


def _torch_ref(input_tensor, mat1, meta, mat2, *, alpha=1.0, beta=1.0):
    grouped = mat1.view(mat1.shape[0], meta.shape[1], 4)
    keep = torch.stack((meta, meta, ~meta, ~meta), dim=-1)
    masked = torch.where(keep, grouped, torch.zeros_like(grouped)).reshape_as(mat1)
    return alpha * (masked @ mat2) + beta * input_tensor


class SparseAddmmBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ADDMM_SHAPES
        self.shape_desc = "M, N"


@pytest.mark.sparse_semi_structured_addmm
def test_sparse_semi_structured_addmm():
    bench = SparseAddmmBenchmark(
        op_name="sparse_semi_structured_addmm",
        torch_op=_torch_ref,
        gems_op=flag_gems._sparse_semi_structured_addmm,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
