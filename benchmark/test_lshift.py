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


class LshiftBenchmark(base.BinaryPointwiseBenchmark):
    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            dims = tuple(shape)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(dims), "other": list(dims)},
                    params={"shift_min": 0, "shift_max_exclusive": 8},
                    builder_args=(dims,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        torch.manual_seed(0)
        inp = torch.randint(0, 0x00FF, shape, dtype=case.dtype, device="cpu").to(
            self.device
        )
        other = torch.randint(0, 8, shape, dtype=case.dtype, device="cpu").to(
            self.device
        )
        return inp, other


@pytest.mark.lshift
def test_lshift():
    bench = LshiftBenchmark(
        op_name="lshift",
        torch_op=torch.ops.aten.__lshift__.Tensor,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
