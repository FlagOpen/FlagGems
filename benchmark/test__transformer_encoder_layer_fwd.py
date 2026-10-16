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


class TransformerEncoderLayerBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (2, 128, 256, 8),
            (8, 128, 512, 8),
            (4, 512, 1024, 16),
        ]

    def set_more_shapes(self):
        return None


def _case_fn(shape, dtype):
    del dtype
    batch, seq_len, embed_dim, num_heads = shape
    for use_gelu in (True, False):
        for norm_first in (True, False):
            yield base.BenchmarkCasePlan(
                shape={"src": [batch, seq_len, embed_dim]},
                params={
                    "num_heads": num_heads,
                    "use_gelu": use_gelu,
                    "norm_first": norm_first,
                },
                builder_args=(tuple(shape), use_gelu, norm_first),
            )


def _build_inputs_fn(plan, dtype, device):
    shape, use_gelu, norm_first = plan.builder_args
    batch, seq_len, embed_dim, num_heads = shape
    del batch, seq_len
    torch.manual_seed(0)
    scale = embed_dim**-0.5
    src = torch.randn(shape[:3], dtype=dtype, device=device)
    qkv_weight = torch.randn(3 * embed_dim, embed_dim, dtype=dtype, device=device) * scale
    qkv_bias = torch.randn(3 * embed_dim, dtype=dtype, device=device) * scale
    proj_weight = torch.randn(embed_dim, embed_dim, dtype=dtype, device=device) * scale
    proj_bias = torch.randn(embed_dim, dtype=dtype, device=device) * scale
    norm_weight_1 = torch.randn(embed_dim, dtype=dtype, device=device)
    norm_bias_1 = torch.randn(embed_dim, dtype=dtype, device=device)
    norm_weight_2 = torch.randn(embed_dim, dtype=dtype, device=device)
    norm_bias_2 = torch.randn(embed_dim, dtype=dtype, device=device)
    ffn_weight_1 = torch.randn(4 * embed_dim, embed_dim, dtype=dtype, device=device) * scale
    ffn_bias_1 = torch.randn(4 * embed_dim, dtype=dtype, device=device) * scale
    ffn_weight_2 = torch.randn(embed_dim, 4 * embed_dim, dtype=dtype, device=device) * scale
    ffn_bias_2 = torch.randn(embed_dim, dtype=dtype, device=device) * scale
    return (
        src,
        embed_dim,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        use_gelu,
        norm_first,
        1e-5,
        norm_weight_1,
        norm_bias_1,
        norm_weight_2,
        norm_bias_2,
        ffn_weight_1,
        ffn_bias_1,
        ffn_weight_2,
        ffn_bias_2,
    )


@pytest.mark.transformer_encoder_layer_fwd
def test_transformer_encoder_layer_fwd():
    bench = TransformerEncoderLayerBenchmark(
        op_name="_transformer_encoder_layer_fwd",
        torch_op=torch._transformer_encoder_layer_fwd,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
