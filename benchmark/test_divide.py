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

from . import base, consts, utils


class DivideBenchmark(base.BinaryPointwiseBenchmark):
    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            dims = tuple(shape)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(dims), "other": list(dims)},
                    params={},
                    builder_args=(dims,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        torch.manual_seed(0)
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        other = utils.generate_tensor_input(shape, case.dtype, self.device)
        other = other.sign().masked_fill_(other == 0, 1) * (other.abs() + 0.5)
        return inp, other


@pytest.mark.divide
def test_divide():
    bench = DivideBenchmark(
        op_name="divide",
        torch_op=torch.ops.aten.divide.Tensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
