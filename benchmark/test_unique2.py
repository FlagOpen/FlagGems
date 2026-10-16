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


_UNIQUE2_SHAPES = [(1024,), (4096,), (16384,), (65536,), (262144,)]
_FLAG_COMBOS = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]


class Unique2Benchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_UNIQUE2_SHAPES)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for return_inverse, return_counts in _FLAG_COMBOS:
        yield base.BenchmarkCasePlan(
            shape={"input": list(dims)},
            params={
                "sorted": True,
                "return_inverse": return_inverse,
                "return_counts": return_counts,
            },
            builder_args=(dims, return_inverse, return_counts),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, return_inverse, return_counts = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randint(0, 1000, shape, dtype=torch.int64, device=device).to(dtype)
    return inp, {
        "sorted": True,
        "return_inverse": return_inverse,
        "return_counts": return_counts,
    }


@pytest.mark.unique2
def test_unique2():
    bench = Unique2Benchmark(
        op_name="unique2",
        torch_op=torch._unique2,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.INT_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()
