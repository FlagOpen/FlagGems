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
from .generated_operator_utils import OperatorBenchmark

SPDIAGS_SHAPES = [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]


def _case_fn(shape, dtype):
    del dtype
    nrows, ncols = shape
    yield base.BenchmarkCasePlan(
        shape={"diagonals": [3, min(nrows, ncols)], "offsets": [3], "output": [nrows, ncols]},
        params={"offsets": [-1, 0, 1], "layout": "sparse_coo"},
        builder_args=(nrows, ncols),
    )


def _build_inputs_fn(plan, dtype, device):
    nrows, ncols = plan.builder_args
    torch.manual_seed(0)
    diagonals = torch.randn((3, min(nrows, ncols)), dtype=dtype, device=device)
    offsets = torch.tensor([-1, 0, 1], dtype=torch.int64, device=device)
    return diagonals.cpu(), offsets.cpu(), [nrows, ncols], diagonals, offsets


def _torch_ref(diagonals, offsets, shape, _device_diagonals, _device_offsets):
    return torch.ops.aten._spdiags(diagonals, offsets, shape)


def _gems_ref(_cpu_diagonals, _cpu_offsets, shape, diagonals, offsets):
    return flag_gems.spdiags(diagonals, offsets, shape)


class SpdiagsBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPDIAGS_SHAPES
        self.shape_desc = "rows, cols"


@pytest.mark.spdiags
def test_spdiags():
    bench = SpdiagsBenchmark(
        op_name="spdiags",
        torch_op=_torch_ref,
        gems_op=_gems_ref,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
