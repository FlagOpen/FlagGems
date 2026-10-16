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


class PoissonNllLossBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = [
        (64, 64),
        (256, 256),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ]

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "target": list(dims)},
        params={"log_input": True, "full": False, "eps": 1e-8, "reduction": 1},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randint(0, 5, shape, device=device).to(dtype)
    return inp, target, True, False, 1e-8, 1


@pytest.mark.poisson_nll_loss
def test_poisson_nll_loss():
    bench = PoissonNllLossBenchmark(
        op_name="poisson_nll_loss",
        torch_op=torch.ops.aten.poisson_nll_loss,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
