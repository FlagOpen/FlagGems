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


SHAPES = [
    ([5, 7, 4], 3),
    ([16, 8, 12, 20], 8),
    ([64, 128, 256], 16),
    ([128, 64, 192, 256], 32),
]


class NestedSelectBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES
        self.shape_desc = "component_sizes, dense_dim"

    def get_case_iter(self, dtype):
        for ordinal, (sizes, dense_dim) in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={
                        "component_sizes": list(sizes),
                        "dense_dim": dense_dim,
                    },
                    params={"dim": 0, "index": 1},
                    builder_args=(tuple(sizes), dense_dim, 1),
                ),
            )

    def build_inputs(self, case):
        torch.manual_seed(0)
        plan = case.builder_args[0]
        sizes, dense_dim, index = plan.builder_args
        components = [
            torch.randn(size, dense_dim, dtype=case.dtype, device=self.device)
            for size in sizes
        ]
        nested = torch.nested.nested_tensor(components, layout=torch.jagged)
        grad = torch.randn(
            sizes[index], dense_dim, dtype=case.dtype, device=self.device
        )
        return grad, nested, 0, index

    def get_tflops(self, op, *args, **kwargs):
        return 0.0


@pytest.mark.nested_select_backward
def test_nested_select_backward():
    NestedSelectBackwardBenchmark(
        op_name="nested_select_backward",
        torch_op=torch.ops.aten._nested_select_backward,
        gems_op=flag_gems._nested_select_backward,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
