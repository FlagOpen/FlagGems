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

import math

import pytest
import torch

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


class NativeMultiHeadAttentionBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (2, 128, 64, 4),
            (4, 256, 64, 8),
            (8, 256, 128, 8),
        ]

    def set_more_shapes(self):
        return None


def _case_fn(shape, dtype):
    del dtype
    batch, seq_len, embed_dim, num_heads = shape
    modes = [
        ("self", None),
        ("encoder_decoder", None),
        ("generic", None),
        ("self", 0),
        ("self", 1),
        ("self", 2),
    ]
    for mode, mask_type in modes:
        yield base.BenchmarkCasePlan(
            shape={
                "query": [batch, seq_len, embed_dim],
                "key": [batch, seq_len, embed_dim],
                "value": [batch, seq_len, embed_dim],
            },
            params={
                "mode": mode,
                "mask_type": mask_type,
                "need_weights": True,
                "average_attn_weights": True,
                "num_heads": num_heads,
            },
            builder_args=(tuple(shape), mode, mask_type),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, mode, mask_type = plan.builder_args
    batch, seq_len, embed_dim, num_heads = shape
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(embed_dim)
    query = torch.randn(batch, seq_len, embed_dim, dtype=dtype, device=device)
    key = torch.randn(batch, seq_len, embed_dim, dtype=dtype, device=device)
    value = torch.randn(batch, seq_len, embed_dim, dtype=dtype, device=device)
    if mode == "self":
        key = value = query
    elif mode == "encoder_decoder":
        value = key
    qkv_weight = torch.randn(3 * embed_dim, embed_dim, dtype=dtype, device=device) * scale
    qkv_bias = torch.randn(3 * embed_dim, dtype=dtype, device=device) * scale
    proj_weight = torch.randn(embed_dim, embed_dim, dtype=dtype, device=device) * scale
    proj_bias = torch.randn(embed_dim, dtype=dtype, device=device) * scale
    mask = None
    if mask_type == 0:
        mask = torch.rand(seq_len, seq_len, device=device) > 0.5
        mask[..., 0] = False
    elif mask_type == 1:
        mask = torch.rand(batch, seq_len, device=device) > 0.5
        mask[:, 0] = False
    elif mask_type == 2:
        mask = torch.rand(batch, num_heads, seq_len, seq_len, device=device) > 0.5
        mask[..., 0] = False
    return (
        query,
        key,
        value,
        embed_dim,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask,
        True,
        True,
        mask_type,
    )


@pytest.mark.native_multi_head_attention
def test_native_multi_head_attention():
    bench = NativeMultiHeadAttentionBenchmark(
        op_name="native_multi_head_attention",
        torch_op=torch._native_multi_head_attention,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
