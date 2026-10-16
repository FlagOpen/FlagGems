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
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (8, 16, 32),
    (16, 32, 64),
    (32, 64, 128),
]


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    if not dims:
        return
    split_extent = dims[0] if len(dims) == 1 else dims[1]
    if split_extent % 4 == 0:
        sections = [4]
    elif split_extent % 2 == 0:
        sections = [2]
    else:
        sections = [1]
    if (
        base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
        and split_extent % 2 == 0
        and 2 not in sections
    ):
        sections.append(2)
    for count in sections:
        if split_extent % count:
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


class HsplitBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SHAPES)


@pytest.mark.hsplit
def test_hsplit():
    bench = HsplitBenchmark(
        op_name="hsplit",
        torch_op=torch.ops.aten.hsplit.int,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
