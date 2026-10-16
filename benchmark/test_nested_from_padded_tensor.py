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
from torch.nested._internal.nested_tensor import _nt_view_dummy

import flag_gems

from . import base, consts


SHAPES = [(8, 16, 16), (32, 32, 64), (128, 64, 256), (512, 128, 64)]


class NestedFromPaddedTensorBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES
        self.shape_desc = "B, T, D"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            batch, padded_length, dense_dim = shape
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"padded": [batch, padded_length, dense_dim]},
                    params={"ragged_idx": 1},
                    builder_args=(batch, padded_length, dense_dim),
                ),
            )

    def build_inputs(self, case):
        torch.manual_seed(0)
        plan = case.builder_args[0]
        batch, padded_length, dense_dim = plan.builder_args
        padded = torch.randn(
            batch,
            padded_length,
            dense_dim,
            dtype=case.dtype,
            device=self.device,
        )
        lengths = [1 + ((index * 7 + 3) % padded_length) for index in range(batch)]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        offsets = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        return padded, offsets, _nt_view_dummy(), 1, None, None, None


@pytest.mark.nested_from_padded_tensor
def test_nested_from_padded_tensor():
    NestedFromPaddedTensorBenchmark(
        op_name="nested_from_padded_tensor",
        torch_op=torch.ops.aten._nested_from_padded_tensor,
        gems_op=flag_gems._nested_from_padded_tensor,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
