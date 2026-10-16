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


SHAPES = [(8, 32), (32, 128), (128, 256), (512, 512), (1024, 1024)]


class NestedViewFromJaggedBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES
        self.shape_desc = "num_batch, max_seq_len"

    def get_case_iter(self, dtype):
        for ordinal, (num_batch, max_seq_len) in enumerate(self.shapes):
            lengths = [
                1 + ((index * 17 + 5) % max_seq_len) for index in range(num_batch)
            ]
            total = sum(lengths)
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={
                        "values": [total, 8],
                        "offsets": [num_batch + 1],
                    },
                    params={"ragged_idx": 1},
                    builder_args=(tuple(lengths), total),
                ),
            )

    def build_inputs(self, case):
        torch.manual_seed(0)
        plan = case.builder_args[0]
        lengths, total = plan.builder_args
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        values = torch.randn(total, 8, dtype=case.dtype, device=self.device)
        offsets = torch.tensor(offsets, dtype=torch.int64, device=self.device)
        return values, offsets, _nt_view_dummy(), None, 1, None, None

    def get_gbps(self, args, latency=None):
        return 0.0

    def get_tflops(self, op, *args, **kwargs):
        return 0.0


@pytest.mark.nested_view_from_jagged
def test_nested_view_from_jagged():
    NestedViewFromJaggedBenchmark(
        op_name="nested_view_from_jagged",
        torch_op=torch.ops.aten._nested_view_from_jagged,
        gems_op=flag_gems._nested_view_from_jagged,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
