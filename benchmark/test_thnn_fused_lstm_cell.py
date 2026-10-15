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

# Shapes for LSTM cell: (batch, hidden_size) where gates = (batch, 4*hidden_size)
LSTM_SHAPES = [
    (1, 4),
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
]


class LSTMCellBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LSTM_SHAPES

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            batch_size, hidden_size = shape
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input_gates": [batch_size, 4 * hidden_size]},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        batch_size, hidden_size = shape
        cur_dtype = case.dtype
        input_gates = torch.randn(
            batch_size, 4 * hidden_size, dtype=cur_dtype, device=self.device
        )
        hidden_gates = torch.randn(
            batch_size, 4 * hidden_size, dtype=cur_dtype, device=self.device
        )
        cx = torch.randn(batch_size, hidden_size, dtype=cur_dtype, device=self.device)
        return input_gates, hidden_gates, cx


@pytest.mark.thnn_fused_lstm_cell
def test_thnn_fused_lstm_cell():
    bench = LSTMCellBenchmark(
        op_name="thnn_fused_lstm_cell",
        torch_op=torch.ops.aten._thnn_fused_lstm_cell,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
