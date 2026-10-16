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


LU_SHAPES = [(64, 64), (256, 128), (512, 512), (768, 768), (1024, 1024)]


class LuUnpackOutBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LU_SHAPES
        self.shape_desc = "M, N"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={
                        "LU_data": list(shape),
                        "LU_pivots": [min(shape)],
                        "P": [shape[0], shape[0]],
                        "L": [shape[0], min(shape)],
                        "U": [min(shape), shape[1]],
                    },
                    params={"unpack_data": True, "unpack_pivots": True},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        m, n = shape
        k = min(shape)
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        lu, pivots = torch.linalg.lu_factor(matrix)
        return lu, pivots, {
            "unpack_data": True,
            "unpack_pivots": True,
            "P": torch.empty((m, m), dtype=case.dtype, device=self.device),
            "L": torch.empty((m, k), dtype=case.dtype, device=self.device),
            "U": torch.empty((k, n), dtype=case.dtype, device=self.device),
        }


@pytest.mark.lu_unpack_out
def test_lu_unpack_out():
    LuUnpackOutBenchmark(
        op_name="lu_unpack_out",
        torch_op=torch.ops.aten.lu_unpack.out,
        dtypes=[torch.float32, torch.float64],
        fresh_inputs=True,
    ).run()
