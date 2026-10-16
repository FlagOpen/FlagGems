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


_DEFAULT_SHAPES = [
    (64, 128, 32),
    (128, 256, 64),
    (256, 512, 128),
    (512, 1024, 256),
]


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    if len(dims) < 3:
        return
    sections = [2 if dims[2] % 2 == 0 else 1]
    if (
        base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
        and dims[2] % 4 == 0
        and 4 not in sections
    ):
        sections.append(4)
    for count in sections:
        if dims[2] % count:
            continue
        yield base.BenchmarkCasePlan(
            shape={"input": list(dims)},
            params={"overload": "int", "sections": count},
            builder_args=(dims, count),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, sections = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, sections


class DsplitBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SHAPES)


@pytest.mark.dsplit
def test_dsplit():
    bench = DsplitBenchmark(
        op_name="dsplit",
        torch_op=torch.ops.aten.dsplit.int,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
