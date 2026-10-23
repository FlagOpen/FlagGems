# Copyright 2026, The FlagOS Contributors.
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

# quantized_rnn_tanh_cell is delegated to the aten quantized cell (the
# fbgemm CPU SIMD kernel behind fbgemm_linear_int8_weight_fp32_activation
# is the only registered backend, and SimpleCell has no is_cuda branch -
# a CUDA-resident activation goes straight into the host-pointer reads and
# segfaults). The native baseline is therefore skipped via the skip_native
# mark; only the FlagGems wrapper latency is measured. The inputs are
# fp32 activations (input (batch, K), hx (batch, hidden)); the weight
# tensors are dim-checked only - the data lives in the opaque fbgemm
# packed objects, which must be CPU-resident products of
# fbgemm_pack_quantized_matrix.
# DTYPE is fixed to fp32 because the activations are fp32-only (the fbgemm
# kernel rejects other activation dtypes).
DTYPE = [torch.float32]


class BenchmarkQuantizedRnnTanhCell(base.Benchmark):
    """
    Benchmark quantized_rnn_tanh_cell operator
    """

    # Shapes are (batch, K_in, hidden): one RNN tanh cell step over a
    # quantized weight pair [hidden, K_in] / [hidden, hidden] (single
    # gate, so the weights have no 3x/4x gate multiplier). Fixed list
    # because the cell has no natural runtime shape parameter beyond the
    # batch and the two feature sizes.
    DEFAULT_SHAPES = [
        (64, 64, 64),
        (256, 256, 128),
        (1024, 512, 256),
        (1024, 1024, 256),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        torch.manual_seed(7)
        for batch, k_in, hidden in self.shapes:
            w_ih = torch.randn(hidden, k_in, dtype=torch.float32)
            w_hh = torch.randn(hidden, hidden, dtype=torch.float32)
            qw_ih, col_ih, scale_ih, zp_ih = (
                torch.ops.aten.fbgemm_linear_quantize_weight(w_ih)
            )
            qw_hh, col_hh, scale_hh, zp_hh = (
                torch.ops.aten.fbgemm_linear_quantize_weight(w_hh)
            )
            packed_ih = torch.ops.aten.fbgemm_pack_quantized_matrix(qw_ih)
            packed_hh = torch.ops.aten.fbgemm_pack_quantized_matrix(qw_hh)
            b_ih = torch.randn(hidden, dtype=torch.float32)
            b_hh = torch.randn(hidden, dtype=torch.float32)
            input = torch.randn(batch, k_in, dtype=torch.float32, device="cuda")
            hx = torch.randn(batch, hidden, dtype=torch.float32, device="cuda")
            yield (
                input,
                hx,
                qw_ih.cuda(),
                qw_hh.cuda(),
                b_ih.cuda(),
                b_hh.cuda(),
                packed_ih,
                packed_hh,
                col_ih.cuda(),
                col_hh.cuda(),
                scale_ih,
                scale_hh,
                zp_ih,
                zp_hh,
            )


@pytest.mark.quantized_rnn_tanh_cell
@pytest.mark.skip_native(
    vendors=["nvidia"],
    reason="aten::quantized_rnn_tanh_cell has no usable CUDA backend: "
    "SimpleCell has no is_cuda branch, so a CUDA input goes straight "
    "into the fbgemm CPU host-pointer reads and segfaults",
)
def test_quantized_rnn_tanh_cell():
    bench = BenchmarkQuantizedRnnTanhCell(
        op_name="quantized_rnn_tanh_cell",
        torch_op=torch.ops.aten.quantized_rnn_tanh_cell,
        gems_op=flag_gems.quantized_rnn_tanh_cell,
        dtypes=DTYPE,
    )
    bench.run()
