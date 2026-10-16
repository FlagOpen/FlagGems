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

SHAPES = [
    (1, 1, 16, 8),
    (2, 3, 32, 64),
    (4, 8, 64, 64),
]


class FillDropoutMaskBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "B, H, Q, K"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for seed, offset in ((42, 0), (7, 1024)):
        yield base.BenchmarkCasePlan(
            shape={"mask": list(dims)},
            params={"dropout_p": 0.0, "seed": seed, "offset": offset},
            builder_args=(dims, seed, offset),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, seed, offset = plan.builder_args
    mask = torch.empty(shape, dtype=dtype, device=device)
    return mask, 0.0, seed, offset


@pytest.mark.fill_mem_eff_dropout_mask_
def test_fill_mem_eff_dropout_mask_():
    FillDropoutMaskBenchmark(
        op_name="fill_mem_eff_dropout_mask_",
        torch_op=torch.ops.aten._fill_mem_eff_dropout_mask_,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
        fresh_inputs=True,
    ).run()
