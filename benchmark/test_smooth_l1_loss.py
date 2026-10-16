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


class SmoothL1LossBenchmark(OperatorBenchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [(262144,), (1024, 1024), (4096, 4096), (64, 512, 512)]
    DEFAULT_SHAPE_DESC = "(B), M, N"

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    reductions = (1,) if base.Config.bench_level != consts.BenchLevel.COMPREHENSIVE else (0, 1, 2)
    for reduction in reductions:
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims), "target": list(dims)},
            params={"reduction": reduction, "beta": 1.0},
            builder_args=(dims, reduction),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, reduction = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randn(shape, dtype=dtype, device=device)
    return inp, target, reduction, 1.0


@pytest.mark.smooth_l1_loss
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_smooth_l1_loss():
    bench = SmoothL1LossBenchmark(
        op_name="smooth_l1_loss",
        torch_op=torch.ops.aten.smooth_l1_loss,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
