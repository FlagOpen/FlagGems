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


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "other": list(dims)},
        params={"domain": "positive"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=device) + 1.0
    other = torch.rand(shape, dtype=dtype, device=device) + 1.0
    return inp, other


class IgammaBenchmark(base.BinaryPointwiseBenchmark, OperatorBenchmark):
    """Keep binary shape and metric policy with generated case listing."""

    def supports_cases(self):
        return True

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.igamma_
def test_igamma_():
    bench = IgammaBenchmark(
        op_name="igamma_",
        torch_op=lambda a, b: a.igamma_(b),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
