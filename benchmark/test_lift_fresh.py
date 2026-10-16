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


SHAPES = [(2, 3), (128, 256), (512, 512)]


class LiftFreshBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES
        self.shape_desc = "M, N"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"self": list(shape)},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        torch.manual_seed(0)
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        return (torch.randn(shape, dtype=case.dtype, device=self.device),)


@pytest.mark.lift_fresh
def test_lift_fresh():
    LiftFreshBenchmark(
        op_name="lift_fresh",
        torch_op=torch.ops.aten.lift_fresh,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
