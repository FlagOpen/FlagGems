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

MATMUL_BACKWARD_SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]


def _baseline(grad, self_tensor, other_tensor, mask):
    grad_self = None
    grad_other = None
    if mask[0]:
        grad_self = torch.matmul(grad, other_tensor.transpose(-2, -1))
    if mask[1]:
        grad_other = torch.matmul(self_tensor.transpose(-2, -1), grad)
    return grad_self, grad_other


class MatmulBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(MATMUL_BACKWARD_SHAPES)
        self.shape_desc = "(M, K, N) for self[M,K] @ other[K,N]"


def _case_fn(shape, dtype):
    del dtype
    m, k, n = map(int, shape)
    yield base.BenchmarkCasePlan(
        shape={"grad": [m, n], "self": [m, k], "other": [k, n]},
        params={"mask": [True, True]},
        builder_args=(m, k, n),
    )


def _build_inputs_fn(plan, dtype, device):
    m, k, n = plan.builder_args
    torch.manual_seed(0)
    grad = torch.randn((m, n), dtype=dtype, device=device)
    self_tensor = torch.randn((m, k), dtype=dtype, device=device)
    other_tensor = torch.randn((k, n), dtype=dtype, device=device)
    return grad, self_tensor, other_tensor, [True, True]


@pytest.mark.matmul_backward
def test_matmul_backward():
    MatmulBackwardBenchmark(
        op_name="matmul_backward",
        torch_op=_baseline,
        gems_op=flag_gems.matmul_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32, torch.float16],
    ).run()
