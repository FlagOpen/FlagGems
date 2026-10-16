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

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import accuracy_utils as utils
from .conftest import TO_CPU


def _make_qkv(batch, num_heads, q_len, kv_len, head_dim, dtype):
    torch.manual_seed(0)
    set_philox_state(1234567890, 0, torch_device_fn.current_device())
    query = torch.empty(
        batch, num_heads, q_len, head_dim, dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05)
    key = torch.empty(
        batch, num_heads, kv_len, head_dim, dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05)
    value = torch.empty(
        batch, num_heads, kv_len, head_dim, dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05)
    return query, key, value


CASES = [
    (2, 4, 512, 512),
    (1, 2, 512, 1024),
    (1, 1, 64, 64),
    (4, 8, 128, 128),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cudnn_attention_forward
@pytest.mark.parametrize("batch,num_heads,q_len,kv_len", CASES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("has_attn_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cudnn_attention_forward(
    batch, num_heads, q_len, kv_len, head_dim, is_causal, has_attn_bias, dtype
):
    if TO_CPU:
        pytest.skip("_cudnn_attention_forward is CUDA-only")
    if is_causal and has_attn_bias:
        pytest.skip("the cuDNN reference rejects causal attention with explicit bias")

    query, key, value = _make_qkv(
        batch, num_heads, q_len, kv_len, head_dim, dtype
    )
    attn_bias = None
    if has_attn_bias:
        attn_bias = torch.randn(
            batch,
            num_heads,
            q_len,
            kv_len,
            dtype=dtype,
            device=flag_gems.device,
        ) * 0.1
    scale = 1.0 / math.sqrt(head_dim)
    args = (
        query,
        key,
        value,
        attn_bias,
        None,
        None,
        q_len,
        kv_len,
        True,
        0.0,
        is_causal,
        False,
    )
    ref_args = (
        utils.to_reference(query, True),
        utils.to_reference(key, True),
        utils.to_reference(value, True),
        None if attn_bias is None else utils.to_reference(attn_bias, True),
        None,
        None,
        q_len,
        kv_len,
        True,
        0.0,
        is_causal,
        False,
    )
    reference = torch.ops.aten._cudnn_attention_forward(*ref_args, scale=scale)
    result = flag_gems.ops.cudnn_attention_forward(*args, scale=scale)

    assert len(result) == len(reference) == 9
    utils.gems_assert_close(
        result[0], reference[0], dtype, equal_nan=True, atol=1e-3
    )
    utils.gems_assert_close(
        result[1].float(), reference[1].float(), torch.float32, equal_nan=True, atol=0.1
    )
    assert result[2] is reference[2] is None
    assert result[3] is reference[3] is None
    assert result[4] == reference[4] == q_len
    assert result[5] == reference[5] == kv_len
    for index in (6, 7):
        assert result[index].shape == reference[index].shape
        assert result[index].dtype == reference[index].dtype
    assert result[8].shape == reference[8].shape
    assert result[8].dtype == reference[8].dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cudnn_attention_forward
def test_cudnn_attention_forward_dropout_rejected():
    if TO_CPU:
        pytest.skip("_cudnn_attention_forward is CUDA-only")
    query, key, value = _make_qkv(1, 2, 64, 64, 64, torch.float16)
    with pytest.raises(NotImplementedError):
        flag_gems.ops.cudnn_attention_forward(
            query,
            key,
            value,
            None,
            None,
            None,
            64,
            64,
            True,
            0.1,
            False,
            False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cudnn_attention_forward
@pytest.mark.parametrize("pass_q,pass_k", [(True, False), (False, True), (True, True)])
def test_cudnn_attention_forward_varlen_rejected(pass_q, pass_k):
    if TO_CPU:
        pytest.skip("_cudnn_attention_forward is CUDA-only")
    query, key, value = _make_qkv(1, 2, 64, 64, 64, torch.float16)
    cum_seq = torch.tensor([0, 64], dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(NotImplementedError):
        flag_gems.ops.cudnn_attention_forward(
            query,
            key,
            value,
            None,
            cum_seq if pass_q else None,
            cum_seq if pass_k else None,
            64,
            64,
            True,
            0.0,
            False,
            False,
        )
