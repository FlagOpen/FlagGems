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
    cfg.TO_CPU, reason="CUDA-only op; no CPU implementation is registered"
)

MHA_SHAPES = [
    (2, 8, 64, 4),
    (4, 16, 128, 8),
    (2, 100, 64, 4),
    (3, 5, 32, 2),
    (2, 8, 40, 5),
]

ATTENTION_ATOL = {
    torch.float32: 1e-4,
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}


def _make_inputs(batch, seq_len, embed_dim, dtype):
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(embed_dim)
    query = torch.randn(batch, seq_len, embed_dim, dtype=dtype, device=flag_gems.device)
    key = torch.randn(batch, seq_len, embed_dim, dtype=dtype, device=flag_gems.device)
    value = torch.randn(batch, seq_len, embed_dim, dtype=dtype, device=flag_gems.device)
    qkv_weight = torch.randn(
        3 * embed_dim, embed_dim, dtype=dtype, device=flag_gems.device
    ) * scale
    qkv_bias = torch.randn(3 * embed_dim, dtype=dtype, device=flag_gems.device) * scale
    proj_weight = torch.randn(
        embed_dim, embed_dim, dtype=dtype, device=flag_gems.device
    ) * scale
    proj_bias = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device) * scale
    return query, key, value, qkv_weight, qkv_bias, proj_weight, proj_bias


def _to_reference(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return utils.to_reference(value, True)
        return utils.to_reference(value, False)
    return value


def _run(
    query,
    key,
    value,
    embed_dim,
    num_heads,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    mask=None,
    need_weights=True,
    average_attn_weights=True,
    mask_type=None,
):
    args = (
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
        need_weights,
        average_attn_weights,
        mask_type,
    )
    ref = torch.ops.aten._native_multi_head_attention(
        *tuple(_to_reference(value) for value in args)
    )
    result = flag_gems._native_multi_head_attention(*args)
    return result, ref


def _assert_result(result, reference, dtype, batch, seq_len, embed_dim, num_heads, average):
    result_out, result_weights = result
    reference_out, reference_weights = reference
    assert result_out.shape == (batch, seq_len, embed_dim)
    assert result_out.dtype == dtype
    utils.gems_assert_close(
        result_out,
        reference_out,
        dtype,
        reduce_dim=seq_len,
        atol=ATTENTION_ATOL[dtype],
    )
    if reference_weights is None:
        assert result_weights is None
        return
    expected_shape = (
        (batch, seq_len, seq_len)
        if average
        else (batch, num_heads, seq_len, seq_len)
    )
    assert result_weights.shape == expected_shape
    assert result_weights.dtype == dtype
    utils.gems_assert_close(
        result_weights,
        reference_weights,
        dtype,
        reduce_dim=seq_len,
        atol=ATTENTION_ATOL[dtype],
    )


@pytest.mark.native_multi_head_attention
@pytest.mark.parametrize("shape", MHA_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("mode", ["self", "encoder_decoder", "generic"])
def test_native_multi_head_attention_input_alias_modes(monkeypatch, shape, dtype, mode):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    batch, seq_len, embed_dim, num_heads = shape
    inputs = _make_inputs(batch, seq_len, embed_dim, dtype)
    query, key, value, *weights = inputs
    if mode == "self":
        key = value = query
    elif mode == "encoder_decoder":
        value = key
    result, reference = _run(
        query, key, value, embed_dim, num_heads, *weights
    )
    _assert_result(
        result, reference, dtype, batch, seq_len, embed_dim, num_heads, True
    )


@pytest.mark.native_multi_head_attention
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("need_weights,average_attn_weights", [(False, True), (True, False)])
def test_native_multi_head_attention_output_options(
    monkeypatch, dtype, need_weights, average_attn_weights
):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    batch, seq_len, embed_dim, num_heads = 2, 8, 64, 4
    query, _, _, *weights = _make_inputs(batch, seq_len, embed_dim, dtype)
    result, reference = _run(
        query,
        query,
        query,
        embed_dim,
        num_heads,
        *weights,
        need_weights=need_weights,
        average_attn_weights=average_attn_weights,
    )
    _assert_result(
        result,
        reference,
        dtype,
        batch,
        seq_len,
        embed_dim,
        num_heads,
        average_attn_weights,
    )


@pytest.mark.native_multi_head_attention
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("mask_type", [0, 1, 2])
def test_native_multi_head_attention_masks(monkeypatch, dtype, mask_type):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    batch, seq_len, embed_dim, num_heads = 2, 8, 64, 4
    query, key, value, *weights = _make_inputs(batch, seq_len, embed_dim, dtype)
    if mask_type == 0:
        mask_shape = (seq_len, seq_len)
    elif mask_type == 1:
        mask_shape = (batch, seq_len)
    else:
        mask_shape = (batch, num_heads, seq_len, seq_len)
    mask = torch.rand(mask_shape, device=flag_gems.device) > 0.5
    if mask_type == 1:
        mask[:, 0] = False
    else:
        mask[..., 0] = False
    result, reference = _run(
        query,
        key,
        value,
        embed_dim,
        num_heads,
        *weights,
        mask=mask,
        mask_type=mask_type,
    )
    _assert_result(
        result, reference, dtype, batch, seq_len, embed_dim, num_heads, True
    )
