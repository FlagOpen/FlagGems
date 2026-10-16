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

from . import accuracy_utils as utils

EFFICIENT_ATTENTION = torch.ops.aten._scaled_dot_product_efficient_attention
CASES = [(1, 2, 8, 16, False, False), (2, 4, 16, 32, True, False), (2, 4, 16, 32, False, True)]


@pytest.mark.scaled_dot_product_efficient_attention
@pytest.mark.parametrize("batch,heads,seq_len,head_dim,is_causal,use_bias", CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scaled_dot_product_efficient_attention(
    batch, heads, seq_len, head_dim, is_causal, use_bias, dtype
):
    torch.manual_seed(0)
    shape = (batch, heads, seq_len, head_dim)
    query = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    key = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    value = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    attn_bias = (
        torch.randn(batch, heads, seq_len, seq_len, dtype=dtype, device=flag_gems.device) * 0.1
        if use_bias
        else None
    )
    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_value = utils.to_reference(value, True)
    ref_bias = utils.to_reference(attn_bias, True)
    reference = EFFICIENT_ATTENTION(
        ref_query, ref_key, ref_value, ref_bias, True, is_causal=is_causal
    )
    with flag_gems.use_gems():
        result = EFFICIENT_ATTENTION(
            query, key, value, attn_bias, True, is_causal=is_causal
        )

    assert len(result) == len(reference) == 4
    utils.gems_assert_close(result[0], reference[0], dtype, atol=3e-2)
    utils.gems_assert_close(result[1], reference[1], torch.float32, atol=3e-2)
    for index in (2, 3):
        assert result[index].shape == reference[index].shape
        assert result[index].dtype == reference[index].dtype


@pytest.mark.scaled_dot_product_efficient_attention
def test_scaled_dot_product_efficient_attention_rejects_dropout():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 8, 16, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(AssertionError, match="dropout_p"):
        EFFICIENT_ATTENTION(query, query, query, None, False, dropout_p=0.1)
