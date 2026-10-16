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

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="ATen FlashAttention is unavailable in CPU reference mode"
)

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [
    (1, 2, 16, 16, 32),
    (2, 4, 17, 31, 64),
    (1, 1, 1, 65, 128),
]


def _make_qkv(batch, heads, q_len, kv_len, head_dim, dtype, *, noncontiguous=False):
    torch.manual_seed(0)
    if noncontiguous:
        query = torch.randn(
            (batch, q_len, heads, head_dim), dtype=dtype, device=flag_gems.device
        ).transpose(1, 2)
        key = torch.randn(
            (batch, kv_len, heads, head_dim), dtype=dtype, device=flag_gems.device
        ).transpose(1, 2)
        value = torch.randn(
            (batch, kv_len, heads, head_dim), dtype=dtype, device=flag_gems.device
        ).transpose(1, 2)
    else:
        query = torch.randn(
            (batch, heads, q_len, head_dim), dtype=dtype, device=flag_gems.device
        )
        key = torch.randn(
            (batch, heads, kv_len, head_dim), dtype=dtype, device=flag_gems.device
        )
        value = torch.randn(
            (batch, heads, kv_len, head_dim), dtype=dtype, device=flag_gems.device
        )
    return query, key, value


def _call(query, key, value, dropout_p, is_causal, return_debug_mask, scale):
    return torch.ops.aten._scaled_dot_product_flash_attention.default(
        query,
        key,
        value,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
    )


def _assert_metadata(actual, expected):
    assert len(actual) == len(expected) == 9
    assert actual[2] is None and expected[2] is None
    assert actual[3] is None and expected[3] is None
    assert actual[4:6] == expected[4:6]
    for index in (6, 7, 8):
        assert actual[index].shape == expected[index].shape
        assert actual[index].dtype == expected[index].dtype


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("scale_mode", ["default", "explicit"])
def test_scaled_dot_product_flash_attention(
    dtype, shape, is_causal, scale_mode
):
    batch, heads, q_len, kv_len, head_dim = shape
    query, key, value = _make_qkv(
        batch, heads, q_len, kv_len, head_dim, dtype
    )
    scale = None if scale_mode == "default" else 0.5 / math.sqrt(head_dim)
    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_value = utils.to_reference(value, True)

    expected = _call(ref_query, ref_key, ref_value, 0.0, is_causal, False, scale)
    with flag_gems.use_gems():
        actual = _call(query, key, value, 0.0, is_causal, False, scale)

    utils.gems_assert_close(actual[0], expected[0], dtype)
    utils.gems_assert_close(actual[1], expected[1], torch.float32)
    _assert_metadata(actual, expected)
    assert actual[0].shape == query.shape
    assert actual[0].untyped_storage().data_ptr() != query.untyped_storage().data_ptr()


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.parametrize("dtype", DTYPES)
def test_scaled_dot_product_flash_attention_noncontiguous(dtype):
    query, key, value = _make_qkv(2, 4, 17, 31, 64, dtype, noncontiguous=True)
    assert not query.is_contiguous()
    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_value = utils.to_reference(value, True)

    expected = _call(ref_query, ref_key, ref_value, 0.0, False, False, None)
    with flag_gems.use_gems():
        actual = _call(query, key, value, 0.0, False, False, None)

    utils.gems_assert_close(actual[0], expected[0], dtype)
    utils.gems_assert_close(actual[1], expected[1], torch.float32)
    _assert_metadata(actual, expected)


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.parametrize("dtype", DTYPES)
def test_scaled_dot_product_flash_attention_debug_mask_metadata(dtype):
    query, key, value = _make_qkv(1, 2, 16, 16, 32, dtype)
    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_value = utils.to_reference(value, True)

    expected = _call(ref_query, ref_key, ref_value, 0.0, False, True, None)
    with flag_gems.use_gems():
        actual = _call(query, key, value, 0.0, False, True, None)

    utils.gems_assert_close(actual[0], expected[0], dtype)
    utils.gems_assert_close(actual[1], expected[1], torch.float32)
    _assert_metadata(actual, expected)
    if actual[8].numel() and expected[8].numel():
        utils.gems_assert_equal(actual[8], expected[8])


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.parametrize("dropout_p", [0.1, 0.25])
def test_scaled_dot_product_flash_attention_dropout_contract(dropout_p):
    query, key, value = _make_qkv(1, 2, 16, 16, 32, torch.float16)
    with flag_gems.use_gems():
        result = _call(query, key, value, dropout_p, False, True, None)

    assert len(result) == 9
    assert result[0].shape == query.shape
    assert result[1].shape == query.shape[:-1]
    assert result[6].numel() > 0
    assert result[8].dtype in (torch.bool, torch.uint8, query.dtype)


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.parametrize(
    "bad_query,bad_key,bad_value",
    [
        ((2, 16, 32), (1, 2, 16, 32), (1, 2, 16, 32)),
        ((1, 2, 16, 32), (1, 3, 16, 32), (1, 2, 16, 32)),
        ((1, 2, 16, 32), (1, 2, 16, 32), (1, 2, 17, 32)),
    ],
)
def test_scaled_dot_product_flash_attention_rejects_invalid_shapes(
    bad_query, bad_key, bad_value
):
    torch.manual_seed(0)
    query = torch.randn(bad_query, dtype=torch.float16, device=flag_gems.device)
    key = torch.randn(bad_key, dtype=torch.float16, device=flag_gems.device)
    value = torch.randn(bad_value, dtype=torch.float16, device=flag_gems.device)
    with pytest.raises((RuntimeError, ValueError)):
        with flag_gems.use_gems():
            _call(query, key, value, 0.0, False, False, None)


@pytest.mark.scaled_dot_product_flash_attention
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_scaled_dot_product_flash_attention_rejects_unsupported_dtype(dtype):
    query = torch.ones((1, 2, 16, 32), dtype=dtype, device=flag_gems.device)
    key = torch.ones_like(query)
    value = torch.ones_like(query)
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        with flag_gems.use_gems():
            _call(query, key, value, 0.0, False, False, None)
