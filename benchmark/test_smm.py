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

SMM_SHAPES = [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 128, 256), (1024, 256, 512)]


def _torch_smm_ref(self, mat):
    return (self.to_dense() @ mat).to_sparse().coalesce()


def _case_fn(shape, dtype):
    del dtype
    m, k, n = shape
    yield base.BenchmarkCasePlan(
        shape={"sparse": [m, k], "dense": [k, n]},
        params={"density": 0.3},
        builder_args=(m, k, n, 0.3),
    )


def _build_inputs_fn(plan, dtype, device):
    m, k, n, density = plan.builder_args
    torch.manual_seed(0)
    dense_source = torch.randn(m, k, dtype=dtype, device=device)
    mask = torch.rand(m, k, device=device) < density
    sparse = (dense_source * mask).to_sparse().coalesce()
    dense = torch.randn(k, n, dtype=dtype, device=device)
    return sparse, dense


class SmmBenchmark(OperatorBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = SMM_SHAPES
        self.shape_desc = "M, K, N"

    def get_tflops(self, op, *args, **kwargs):
        sparse, dense = args[:2]
        return 2 * sparse._nnz() * dense.shape[1]


@pytest.mark.smm
def test_smm():
    bench = SmmBenchmark(
        op_name="smm",
        torch_op=_torch_smm_ref,
        gems_op=flag_gems.smm,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    )
    bench.run()
