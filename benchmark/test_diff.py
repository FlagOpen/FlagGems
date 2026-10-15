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

import functools

import pytest
import torch

import flag_gems

from . import base, consts


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    return (inp,)


class DiffBenchmark(base.GenericBenchmark2DOnly):
    def set_shapes(self, *args, **kwargs):
        super().set_shapes(*args, **kwargs)
        self.shapes = [s for s in self.shapes if all(d >= 2 for d in s)]
        self.shapes += [
            (16,),
            (4096,),
            (64, 128, 256),
            (32, 1024, 1024),
            (8, 4096, 4096),
        ]


@pytest.mark.diff
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_diff():
    bench = DiffBenchmark(
        op_name="diff",
        torch_op=torch.diff,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.diff
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_diff_n2():
    bench = DiffBenchmark(
        op_name="diff",
        torch_op=functools.partial(torch.diff, n=2),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
