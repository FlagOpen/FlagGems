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
import triton

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="ATen FlashAttention is unavailable in CPU reference mode"
)

DTYPES = [torch.float16, torch.bfloat16]
DENSE_CASES = [
    (1, 2, 2, 16, 16, 32),
    (2, 4, 4, 17, 31, 64),
    (1, 4, 2, 33, 65, 128),
]


def _set_seed():
    torch.manual_seed(0)
    set_philox_state(1234567890, 0, torch_device_fn.current_device())


def _make_qkv(batch, q_heads, kv_heads, q_len, kv_len, head_dim, dtype):
    _set_seed()
    query = torch.empty(
        (batch, q_len, q_heads, head_dim), dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05)
    key = torch.empty(
        (batch, kv_len, kv_heads, head_dim), dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05)
    value = torch.empty_like(key).uniform_(-0.05, 0.05)
    return query, key, value


def _call(
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
    **kwargs,
):
    return torch.ops.aten._flash_attention_forward.default(
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
        **kwargs,
    )


def _reference_args(query, key, value, cum_seq_q=None, cum_seq_k=None):
    return (
        utils.to_reference(query, True),
        utils.to_reference(key, True),
        utils.to_reference(value, True),
        utils.to_reference(cum_seq_q, False) if cum_seq_q is not None else None,
        utils.to_reference(cum_seq_k, False) if cum_seq_k is not None else None,
    )


def _assert_all_outputs(actual, expected, dtype):
    assert len(actual) == len(expected) == 5
    utils.gems_assert_close(actual[0], expected[0], dtype)
    utils.gems_assert_close(actual[1], expected[1], torch.float32)
    for index in (2, 3, 4):
        assert actual[index].shape == expected[index].shape
        assert actual[index].dtype == expected[index].dtype


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", DENSE_CASES)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("scale_mode", ["default", "explicit"])
def test__flash_attention_forward(dtype, shape, is_causal, scale_mode):
    batch, q_heads, kv_heads, q_len, kv_len, head_dim = shape
    query, key, value = _make_qkv(
        batch, q_heads, kv_heads, q_len, kv_len, head_dim, dtype
    )
    ref_query, ref_key, ref_value, _, _ = _reference_args(query, key, value)
    scale = None if scale_mode == "default" else 0.5 / math.sqrt(head_dim)

    expected = _call(
        ref_query,
        ref_key,
        ref_value,
        None,
        None,
        q_len,
        kv_len,
        0.0,
        is_causal,
        False,
        scale=scale,
    )
    with flag_gems.use_gems():
        actual = _call(
            query,
            key,
            value,
            None,
            None,
            q_len,
            kv_len,
            0.0,
            is_causal,
            False,
            scale=scale,
        )

    _assert_all_outputs(actual, expected, dtype)
    assert actual[0].shape == query.shape
    assert actual[0].untyped_storage().data_ptr() != query.untyped_storage().data_ptr()


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.skipif(
    flag_gems.vendor_name
    in {"hygon", "metax", "mthreads", "kunlunxin", "cambricon", "tsingmicro"},
    reason="Existing FlashAttention sliding-window backend limitations: issues #2810, #2811, #2812, #2814, #5254, #4083",
)
@pytest.mark.parametrize(
    "window_size_left,window_size_right", [(8, 0), (8, 8), (None, 4)]
)
def test__flash_attention_forward_windowed(window_size_left, window_size_right):
    dtype = torch.float16
    query, key, value = _make_qkv(1, 2, 2, 17, 31, 64, dtype)
    ref_query, ref_key, ref_value, _, _ = _reference_args(query, key, value)
    kwargs = {
        "scale": None,
        "window_size_left": window_size_left,
        "window_size_right": window_size_right,
    }
    expected = _call(
        ref_query, ref_key, ref_value, None, None, 17, 31, 0.0, False, False, **kwargs
    )
    with flag_gems.use_gems():
        actual = _call(
            query, key, value, None, None, 17, 31, 0.0, False, False, **kwargs
        )

    _assert_all_outputs(actual, expected, dtype)


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.skipif(
    flag_gems.vendor_name in {"hygon", "mthreads", "kunlunxin", "tsingmicro"},
    reason="Existing FlashAttention GQA/ALiBi backend limitations: issues #2810, #2812, #2814, #4083",
)
@pytest.mark.parametrize("is_causal", [False, True])
def test__flash_attention_forward_gqa_alibi(is_causal):
    dtype = torch.float16
    query, key, value = _make_qkv(2, 4, 2, 16, 31, 64, dtype)
    alibi_slopes = torch.linspace(
        0.1, 0.4, steps=8, dtype=torch.float32, device=flag_gems.device
    ).reshape(2, 4)
    ref_query, ref_key, ref_value, _, _ = _reference_args(query, key, value)
    ref_alibi = utils.to_reference(alibi_slopes, False)

    expected = _call(
        ref_query,
        ref_key,
        ref_value,
        None,
        None,
        16,
        31,
        0.0,
        is_causal,
        False,
        scale=None,
        alibi_slopes=ref_alibi,
    )
    with flag_gems.use_gems():
        actual = _call(
            query,
            key,
            value,
            None,
            None,
            16,
            31,
            0.0,
            is_causal,
            False,
            scale=None,
            alibi_slopes=alibi_slopes,
        )

    _assert_all_outputs(actual, expected, dtype)


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.parametrize("dtype", DTYPES)
def test__flash_attention_forward_noncontiguous_bshd(dtype):
    _set_seed()
    query = torch.empty(
        (2, 4, 17, 64), dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05).transpose(1, 2)
    key = torch.empty(
        (2, 4, 31, 64), dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05).transpose(1, 2)
    value = torch.empty(
        (2, 4, 31, 64), dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05).transpose(1, 2)
    assert not query.is_contiguous()
    ref_query, ref_key, ref_value, _, _ = _reference_args(query, key, value)

    expected = _call(
        ref_query,
        ref_key,
        ref_value,
        None,
        None,
        17,
        31,
        0.0,
        False,
        False,
        scale=None,
    )
    with flag_gems.use_gems():
        actual = _call(
            query,
            key,
            value,
            None,
            None,
            17,
            31,
            0.0,
            False,
            False,
            scale=None,
        )

    _assert_all_outputs(actual, expected, dtype)


@pytest.mark.underscore_flash_attention_forward
def test__flash_attention_forward_seqused_k_metadata():
    dtype = torch.float16
    query, key, value = _make_qkv(2, 2, 2, 4, 8, 32, dtype)
    seqused_k = torch.tensor([5, 8], dtype=torch.int32, device=flag_gems.device)
    ref_query, ref_key, ref_value, _, _ = _reference_args(query, key, value)
    ref_seqused_k = utils.to_reference(seqused_k, False)

    expected = _call(
        ref_query,
        ref_key,
        ref_value,
        None,
        None,
        4,
        8,
        0.0,
        False,
        False,
        scale=None,
        seqused_k=ref_seqused_k,
    )
    with flag_gems.use_gems():
        actual = _call(
            query,
            key,
            value,
            None,
            None,
            4,
            8,
            0.0,
            False,
            False,
            scale=None,
            seqused_k=seqused_k,
        )

    _assert_all_outputs(actual, expected, dtype)


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.parametrize("dtype", DTYPES)
def test__flash_attention_forward_varlen(dtype):
    _set_seed()
    cum_seq_q = torch.tensor([0, 3, 8], dtype=torch.int32, device=flag_gems.device)
    cum_seq_k = torch.tensor([0, 5, 9], dtype=torch.int32, device=flag_gems.device)
    query = torch.empty((8, 2, 32), dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    key = torch.empty((9, 2, 32), dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    value = torch.empty_like(key).uniform_(-0.05, 0.05)
    ref_query, ref_key, ref_value, ref_cum_q, ref_cum_k = _reference_args(
        query, key, value, cum_seq_q, cum_seq_k
    )

    expected = _call(
        ref_query,
        ref_key,
        ref_value,
        ref_cum_q,
        ref_cum_k,
        5,
        5,
        0.0,
        False,
        False,
        scale=None,
    )
    with flag_gems.use_gems():
        actual = _call(
            query,
            key,
            value,
            cum_seq_q,
            cum_seq_k,
            5,
            5,
            0.0,
            False,
            False,
            scale=None,
        )

    _assert_all_outputs(actual, expected, dtype)


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.skipif(triton.__version__ < "3.1", reason="Dropout requires Triton >= 3.1")
@pytest.mark.skipif(
    flag_gems.vendor_name in {"hygon", "mthreads", "kunlunxin"},
    reason="Existing FlashAttention dropout backend limitations: issues #2810, #2812, #2814",
)
@pytest.mark.parametrize("dropout_p", [0.1, 0.25])
def test__flash_attention_forward_dropout_debug_mask(dropout_p):
    dtype = torch.float16
    query, key, value = _make_qkv(1, 2, 2, 128, 128, 32, dtype)

    _set_seed()
    with flag_gems.use_gems():
        actual = _call(
            query,
            key,
            value,
            None,
            None,
            128,
            128,
            dropout_p,
            False,
            True,
            scale=None,
        )
    _set_seed()
    with flag_gems.use_gems():
        replay = _call(
            query,
            key,
            value,
            None,
            None,
            128,
            128,
            dropout_p,
            False,
            True,
            scale=None,
        )

    assert len(actual) == len(replay) == 5
    utils.gems_assert_equal(actual[0], replay[0])
    utils.gems_assert_equal(actual[1], replay[1])
    for result, repeated in zip(actual[2:], replay[2:]):
        assert result.shape == repeated.shape
        assert result.dtype == repeated.dtype
        if result.numel():
            utils.gems_assert_equal(result, repeated)
    debug_mask = actual[4]
    kept_or_dropped = torch.sum(debug_mask != 0)
    assert kept_or_dropped.item() > 0
    observed_dropout = (
        torch.sum(debug_mask < 0).float() / kept_or_dropped.float()
    ).item()
    assert abs(observed_dropout - dropout_p) < 0.03


@pytest.mark.underscore_flash_attention_forward
def test__flash_attention_forward_rejects_partial_varlen_metadata():
    query, key, value = _make_qkv(1, 2, 2, 16, 16, 32, torch.float16)
    cum_seq_q = torch.tensor([0, 16], dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, ValueError)):
        with flag_gems.use_gems():
            _call(
                query,
                key,
                value,
                cum_seq_q,
                None,
                16,
                16,
                0.0,
                False,
                False,
                scale=None,
            )


@pytest.mark.underscore_flash_attention_forward
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test__flash_attention_forward_rejects_unsupported_dtype(dtype):
    query = torch.ones((1, 16, 2, 32), dtype=dtype, device=flag_gems.device)
    key = torch.ones_like(query)
    value = torch.ones_like(query)
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        with flag_gems.use_gems():
            _call(
                query,
                key,
                value,
                None,
                None,
                16,
                16,
                0.0,
                False,
                False,
                scale=None,
            )
