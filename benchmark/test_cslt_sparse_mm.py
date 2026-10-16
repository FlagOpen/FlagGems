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

from . import base
from .generated_operator_utils import OperatorBenchmark

CSLT_SHAPES = [(64, 128, 64), (128, 256, 128), (256, 512, 256), (512, 1024, 512)]


def _case_fn(shape, dtype):
    del dtype
    m, k, n = shape
    yield base.BenchmarkCasePlan(
        shape={"compressed_A_source": [m, k], "dense_B": [k, n]},
        params={"sparsity": "2:4"},
        builder_args=(m, k, n),
    )


def _build_inputs_fn(plan, dtype, device):
    m, k, n = plan.builder_args
    torch.manual_seed(0)
    source = torch.randn(m, k, dtype=dtype, device=device).view(m, k // 4, 4)
    indices = source.abs().argsort(dim=-1)
    mask = torch.zeros_like(source)
    mask.scatter_(-1, indices[..., 2:], 1.0)
    compressed = torch._cslt_compress((source * mask).view(m, k).contiguous())
    dense = torch.randn(k, n, dtype=dtype, device=device)
    return compressed, dense


class CsltSparseMMBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CSLT_SHAPES
        self.shape_desc = "M, K, N"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the Triton decoder models the Hopper cuSPARSELt metadata layout",
)
@pytest.mark.cslt_sparse_mm
def test_cslt_sparse_mm():
    bench = CsltSparseMMBenchmark(
        op_name="cslt_sparse_mm",
        torch_op=torch._cslt_sparse_mm,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
