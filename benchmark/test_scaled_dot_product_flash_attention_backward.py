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

from . import base
from .generated_operator_utils import OperatorBenchmark

CASES = [
    (1, 2, 128, 128, 64, False, 0.0, "default"),
    (2, 4, 256, 256, 64, True, 0.0, "explicit"),
    (1, 8, 128, 512, 128, False, 0.0, "explicit"),
    (1, 8, 1, 1024, 128, False, 0.0, "default"),
    (1, 2, 128, 128, 64, False, 0.2, "explicit"),
]


class ScaledDotProductFlashAttentionBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = (
            "batch, heads, q_len, kv_len, head_dim, causal, dropout, scale"
        )


def _case_fn(shape, dtype):
    del dtype
    batch, heads, q_len, kv_len, head_dim, is_causal, dropout_p, scale_mode = shape
    yield base.BenchmarkCasePlan(
        shape={
            "grad_out": [batch, q_len, heads, head_dim],
            "query": [batch, q_len, heads, head_dim],
            "key": [batch, kv_len, heads, head_dim],
            "value": [batch, kv_len, heads, head_dim],
        },
        params={
            "dropout_p": dropout_p,
            "is_causal": is_causal,
            "scale_mode": scale_mode,
            "cum_seq_q": None,
            "cum_seq_k": None,
        },
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    batch, heads, q_len, kv_len, head_dim, is_causal, dropout_p, scale_mode = (
        plan.builder_args[0]
    )
    torch.manual_seed(0)
    query = torch.empty(
        (batch, q_len, heads, head_dim), dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    key = torch.empty(
        (batch, kv_len, heads, head_dim), dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    value = torch.empty_like(key).uniform_(-0.05, 0.05)
    grad_out = torch.empty_like(query).normal_()
    scale = None if scale_mode == "default" else 1.0 / math.sqrt(head_dim)
    out, logsumexp, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
        query,
        key,
        value,
        None,
        None,
        q_len,
        kv_len,
        dropout_p,
        is_causal,
        False,
        scale=scale,
    )
    return (
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp.float(),
        None,
        None,
        q_len,
        kv_len,
        dropout_p,
        is_causal,
        rng_state[0],
        rng_state[1],
        {"scale": scale},
    )


@pytest.mark.scaled_dot_product_flash_attention_backward
def test_scaled_dot_product_flash_attention_backward():
    ScaledDotProductFlashAttentionBackwardBenchmark(
        op_name="scaled_dot_product_flash_attention_backward",
        torch_op=torch.ops.aten._scaled_dot_product_flash_attention_backward.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float16, torch.bfloat16],
    ).run()
