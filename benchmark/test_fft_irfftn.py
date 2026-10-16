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

SHAPES = [(8,), (16,), (32,), (64,), (128,), (4, 64), (8, 128), (2, 4, 32), (4, 8, 64)]


class OperatorCaseBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "(*B), M, N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    real = torch.randn(shape, dtype=torch.float32, device=device)
    return (torch.fft.rfftn(real),)


@pytest.mark.fft_irfftn
def test_fft_irfftn():
    OperatorCaseBenchmark(
        op_name="fft_irfftn",
        torch_op=torch.fft.irfftn,
        gems_op=flag_gems.fft_irfftn,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.complex64],
    ).run()
