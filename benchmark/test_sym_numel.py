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

_DEFAULT_SHAPES = [(2, 3), (10, 20, 30), (5, 10), (100,), (1, 2, 3, 4)]


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    return (torch.randn(shape, dtype=dtype, device=device),)


class SymNumelBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SHAPES)
        for shape in _DEFAULT_SHAPES:
            if shape not in self.shapes:
                self.shapes.append(shape)

    def set_more_shapes(self):
        return []


@pytest.mark.sym_numel
def test_sym_numel():
    bench = SymNumelBenchmark(
        op_name="sym_numel",
        torch_op=torch.ops.aten.sym_numel,
        gems_op=flag_gems.sym_numel,
        dtypes=[torch.float32],
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
    )
    bench.run()
