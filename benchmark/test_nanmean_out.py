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


class NanmeanOutBenchmark(base.UnaryReductionBenchmark):
    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        ordinal = 0
        for shape in self.shapes:
            dims = [None]
            if len(shape) >= 2:
                dims.extend([-1, 0])
            if len(shape) >= 3:
                dims.append(1)
            for dim in dims:
                out_shape = [] if dim is None else list(shape)
                if dim is not None:
                    del out_shape[dim % len(shape)]
                yield self._case_from_plan(
                    dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": list(shape), "out": out_shape},
                        params={} if dim is None else {"dim": dim},
                        builder_args=(tuple(shape), dim, tuple(out_shape)),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, dim, out_shape = plan.builder_args
        torch.manual_seed(0)
        inp = torch.randn(shape, dtype=case.dtype, device=self.device) * 10
        inp[torch.rand(shape, device=self.device) > 0.7] = float("nan")
        out = torch.empty(out_shape, dtype=case.dtype, device=self.device)
        if dim is None:
            return inp, {"out": out}
        return inp, dim, {"out": out}


@pytest.mark.nanmean_out
def test_nanmean_out():
    bench = NanmeanOutBenchmark(
        op_name="nanmean_out",
        torch_op=torch.nanmean,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.set_gems(flag_gems.nanmean_out)
    bench.run()
