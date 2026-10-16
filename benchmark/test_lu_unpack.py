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


LU_SHAPES = [(64, 64), (256, 128), (512, 512), (768, 768), (1024, 1024)]


class LuUnpackBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LU_SHAPES
        self.shape_desc = "M, N"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        ordinal = 0
        for shape in self.shapes:
            for unpack_data, unpack_pivots in ((True, True), (False, True), (True, False)):
                yield self._case_from_plan(
                    dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={
                            "LU_data": list(shape),
                            "LU_pivots": [min(shape)],
                        },
                        params={
                            "unpack_data": unpack_data,
                            "unpack_pivots": unpack_pivots,
                        },
                        builder_args=(tuple(shape), unpack_data, unpack_pivots),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, unpack_data, unpack_pivots = plan.builder_args
        torch.manual_seed(0)
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        lu, pivots = torch.linalg.lu_factor(matrix)
        return lu, pivots, {
            "unpack_data": unpack_data,
            "unpack_pivots": unpack_pivots,
        }


@pytest.mark.lu_unpack
def test_lu_unpack():
    LuUnpackBenchmark(
        op_name="lu_unpack",
        torch_op=torch.ops.aten.lu_unpack,
        dtypes=[torch.float32, torch.float64],
    ).run()
