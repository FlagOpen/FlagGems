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

SHAPES = [(1024, 1024), (4096, 256)]
REDUCE_MODES = ("sum", "prod", "mean", "amax", "amin")


class ScatterReduceInplaceBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "M,N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for reduce in REDUCE_MODES:
        for include_self in (True, False):
            yield base.BenchmarkCasePlan(
                shape={"self": list(dims), "index": list(dims), "src": list(dims)},
                params={"dim": 1, "reduce": reduce, "include_self": include_self},
                builder_args=(dims, reduce, include_self),
            )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    shape, reduce, include_self = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    src = torch.randn(shape, dtype=dtype, device=device)
    if reduce == "prod":
        inp = torch.rand(shape, dtype=dtype, device=device) * 0.2 + 0.9
        src = torch.rand(shape, dtype=dtype, device=device) * 0.2 + 0.9
    index = torch.randint(0, shape[1], shape, dtype=torch.long, device=device)
    return inp, 1, index, src, reduce, {"include_self": include_self}


@pytest.mark.scatter_reduce_
def test_scatter_reduce_():
    bench = ScatterReduceInplaceBenchmark(
        op_name="scatter_reduce_",
        torch_op=torch.ops.aten.scatter_reduce_.two,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
