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

CORE_SHAPES = [(1024,), (256, 256), (1024, 1024), (64, 512, 256)]
COMPREHENSIVE_SHAPES = [(2048, 2048), (128, 1024, 512)]


class LinalgPowsumBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(COMPREHENSIVE_SHAPES)
        self.shape_desc = "reduction input shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    dim = None if len(dims) == 1 else [-1]
    for ord_value in (1, 2, 3):
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims)},
            params={"ord": ord_value, "dim": dim, "keepdim": False},
            builder_args=(dims, ord_value, dim),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, ord_value, dim = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    return inp, ord_value, dim, False


@pytest.mark.linalg__powsum
def test_linalg__powsum():
    LinalgPowsumBenchmark(
        op_name="linalg__powsum",
        torch_op=torch.ops.aten.linalg__powsum.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
