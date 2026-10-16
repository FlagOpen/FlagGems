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
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(256, 256, 1024), (1024, 1024, 4096), (1024, 1024, 16384), (4096, 4096, 16384), (4096, 4096, 65536)]
INDEX_DTYPES = [torch.int32, torch.int64]


def _make_csc(shape, index_dtype, device):
    torch.manual_seed(0)
    rows, cols, nnz = shape
    counts = [nnz // cols + (1 if i < nnz % cols else 0) for i in range(cols)]
    ccol = torch.tensor([0] + list(torch.cumsum(torch.tensor(counts), dim=0).tolist()), dtype=index_dtype, device=device)
    pieces = []
    for count in counts:
        pieces.append(torch.arange(count, dtype=index_dtype, device=device) % rows)
    row = torch.cat(pieces) if pieces else torch.empty(0, dtype=index_dtype, device=device)
    values = torch.randn(row.numel(), dtype=torch.float32, device=device)
    return torch.sparse_csc_tensor(ccol, row, values, size=(rows, cols))


class RowIndicesCopyBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "rows, cols, nnz"


def _case_fn(shape, dtype):
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "out": [dims[2]]},
        params={"index_dtype": str(dtype).removeprefix("torch.")},
        builder_args=(dims, False),
    )


def _out_case_fn(shape, dtype):
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "out": [dims[2]]},
        params={"index_dtype": str(dtype).removeprefix("torch.")},
        builder_args=(dims, True),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, _ = plan.builder_args
    return (_make_csc(shape, dtype, device),)


def _out_build_inputs_fn(plan, dtype, device):
    shape, _ = plan.builder_args
    csc = _make_csc(shape, dtype, device)
    return csc, {"out": torch.empty(shape[2], dtype=dtype, device=device)}


@pytest.mark.row_indices_copy
def test_row_indices_copy():
    RowIndicesCopyBenchmark(
        op_name="row_indices_copy",
        torch_op=torch.row_indices_copy,
        gems_op=flag_gems.row_indices_copy,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=INDEX_DTYPES,
    ).run()


@pytest.mark.row_indices_copy_out
def test_row_indices_copy_out():
    RowIndicesCopyBenchmark(
        op_name="row_indices_copy_out",
        torch_op=torch.row_indices_copy,
        gems_op=flag_gems.row_indices_copy_out,
        case_fn=_out_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=INDEX_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
