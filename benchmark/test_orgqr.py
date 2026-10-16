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

from . import base


ORGQR_SHAPES = [(16, 8), (32, 16), (64, 32), (128, 64), (128, 128)]


class OrgqrBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ORGQR_SHAPES
        self.shape_desc = "M, N"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape), "tau": [min(shape)]},
                    params={},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        return torch.geqrf(matrix)


@pytest.mark.orgqr
def test_orgqr():
    OrgqrBenchmark(
        op_name="orgqr",
        torch_op=torch.orgqr,
        gems_op=flag_gems.orgqr,
        dtypes=[torch.float32, torch.float64],
    ).run()


@pytest.mark.orgqr_out
def test_orgqr_out():
    def torch_orgqr_out(inp, tau):
        return torch.orgqr(inp, tau, out=torch.empty_like(inp))

    def gems_orgqr_out(inp, tau):
        return flag_gems.orgqr_out(inp, tau, out=torch.empty_like(inp))

    OrgqrBenchmark(
        op_name="orgqr_out",
        torch_op=torch_orgqr_out,
        gems_op=gems_orgqr_out,
        dtypes=[torch.float32, torch.float64],
    ).run()
