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
    ("dense", 1, 2, 2, 128, 128, 64, False, 0.0, False, None, None),
    ("dense", 2, 4, 4, 256, 256, 64, True, 0.0, False, None, None),
    ("dense", 1, 8, 2, 128, 512, 128, False, 0.0, False, None, None),
    ("dense", 1, 8, 1, 1, 1024, 128, False, 0.0, False, None, None),
    ("dense", 1, 4, 4, 128, 512, 128, False, 0.0, False, 128, 0),
    ("dense", 1, 2, 2, 128, 128, 64, False, 0.2, True, None, None),
    ("varlen", 2, 2, 8, 9, 5, 5, 64, False, 0.0, False, None, None),
]


class FlashAttentionForwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "layout and FlashAttention dimensions/options"


def _case_fn(shape, dtype):
    del dtype
    mode = shape[0]
    if mode == "dense":
        (
            _,
            batch,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            is_causal,
            dropout_p,
            return_debug_mask,
            window_left,
            window_right,
        ) = shape
        tensor_shapes = {
            "query": [batch, q_len, q_heads, head_dim],
            "key": [batch, kv_len, kv_heads, head_dim],
            "value": [batch, kv_len, kv_heads, head_dim],
        }
    else:
        (
            _,
            sequences,
            heads,
            total_q,
            total_k,
            max_q,
            max_k,
            head_dim,
            is_causal,
            dropout_p,
            return_debug_mask,
            window_left,
            window_right,
        ) = shape
        tensor_shapes = {
            "query": [total_q, heads, head_dim],
            "key": [total_k, heads, head_dim],
            "value": [total_k, heads, head_dim],
            "sequences": sequences,
            "max_q": max_q,
            "max_k": max_k,
        }
    yield base.BenchmarkCasePlan(
        shape=tensor_shapes,
        params={
            "mode": mode,
            "is_causal": is_causal,
            "dropout_p": dropout_p,
            "return_debug_mask": return_debug_mask,
            "window_size_left": window_left,
            "window_size_right": window_right,
        },
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    mode = shape[0]
    torch.manual_seed(0)
    if mode == "dense":
        (
            _,
            batch,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            is_causal,
            dropout_p,
            return_debug_mask,
            window_left,
            window_right,
        ) = shape
        query = torch.empty(
            (batch, q_len, q_heads, head_dim), dtype=dtype, device=device
        ).uniform_(-0.05, 0.05)
        key = torch.empty(
            (batch, kv_len, kv_heads, head_dim), dtype=dtype, device=device
        ).uniform_(-0.05, 0.05)
        value = torch.empty_like(key).uniform_(-0.05, 0.05)
        cum_seq_q = None
        cum_seq_k = None
        max_q = q_len
        max_k = kv_len
    else:
        (
            _,
            sequences,
            heads,
            total_q,
            total_k,
            max_q,
            max_k,
            head_dim,
            is_causal,
            dropout_p,
            return_debug_mask,
            window_left,
            window_right,
        ) = shape
        query = torch.empty(
            (total_q, heads, head_dim), dtype=dtype, device=device
        ).uniform_(-0.05, 0.05)
        key = torch.empty(
            (total_k, heads, head_dim), dtype=dtype, device=device
        ).uniform_(-0.05, 0.05)
        value = torch.empty_like(key).uniform_(-0.05, 0.05)
        if sequences != 2:
            raise ValueError("the static varlen benchmark plan expects two sequences")
        cum_seq_q = torch.tensor([0, total_q - max_q, total_q], dtype=torch.int32, device=device)
        cum_seq_k = torch.tensor([0, total_k - max_k, total_k], dtype=torch.int32, device=device)
    kwargs = {"scale": 1.0 / math.sqrt(head_dim)}
    if window_left is not None or window_right is not None:
        kwargs["window_size_left"] = window_left
        kwargs["window_size_right"] = window_right
    return (
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        kwargs,
    )


@pytest.mark.underscore_flash_attention_forward
def test__flash_attention_forward():
    FlashAttentionForwardBenchmark(
        op_name="_flash_attention_forward",
        torch_op=torch.ops.aten._flash_attention_forward.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float16, torch.bfloat16],
    ).run()
