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


def _make_inputs(batch_size, seq_len, embed_dim, dtype):
    torch.manual_seed(0)
    scale = embed_dim**-0.5
    src = torch.randn(
        batch_size, seq_len, embed_dim, dtype=dtype, device=flag_gems.device
    )
    qkv_weight = torch.randn(
        3 * embed_dim, embed_dim, dtype=dtype, device=flag_gems.device
    ) * scale
    qkv_bias = torch.randn(3 * embed_dim, dtype=dtype, device=flag_gems.device) * scale
    proj_weight = torch.randn(
        embed_dim, embed_dim, dtype=dtype, device=flag_gems.device
    ) * scale
    proj_bias = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device) * scale
    norm_weight_1 = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device)
    norm_bias_1 = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device)
    norm_weight_2 = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device)
    norm_bias_2 = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device)
    ffn_weight_1 = torch.randn(
        4 * embed_dim, embed_dim, dtype=dtype, device=flag_gems.device
    ) * scale
    ffn_bias_1 = torch.randn(4 * embed_dim, dtype=dtype, device=flag_gems.device) * scale
    ffn_weight_2 = torch.randn(
        embed_dim, 4 * embed_dim, dtype=dtype, device=flag_gems.device
    ) * scale
    ffn_bias_2 = torch.randn(embed_dim, dtype=dtype, device=flag_gems.device) * scale
    return (
        src,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
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
@pytest.mark.parametrize("shape", [(2, 16, 64, 4), (1, 7, 32, 2)])
@pytest.mark.parametrize("use_gelu", [True, False])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_transformer_encoder_layer_fwd(shape, use_gelu, norm_first, dtype):
    batch_size, seq_len, embed_dim, num_heads = shape
    tensors = _make_inputs(batch_size, seq_len, embed_dim, dtype)
    (
        src,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        norm_weight_1,
        norm_bias_1,
        norm_weight_2,
        norm_bias_2,
        ffn_weight_1,
        ffn_bias_1,
        ffn_weight_2,
        ffn_bias_2,
    ) = tensors
    eps = 1e-5
    ref_tensors = tuple(utils.to_reference(value, True) for value in tensors)
    (
        ref_src,
        ref_qkv_weight,
        ref_qkv_bias,
        ref_proj_weight,
        ref_proj_bias,
        ref_norm_weight_1,
        ref_norm_bias_1,
        ref_norm_weight_2,
        ref_norm_bias_2,
        ref_ffn_weight_1,
        ref_ffn_bias_1,
        ref_ffn_weight_2,
        ref_ffn_bias_2,
    ) = ref_tensors

    reference = torch._transformer_encoder_layer_fwd(
        ref_src,
        embed_dim,
        num_heads,
        ref_qkv_weight,
        ref_qkv_bias,
        ref_proj_weight,
        ref_proj_bias,
        use_gelu,
        norm_first,
        eps,
        ref_norm_weight_1,
        ref_norm_bias_1,
        ref_norm_weight_2,
        ref_norm_bias_2,
        ref_ffn_weight_1,
        ref_ffn_bias_1,
        ref_ffn_weight_2,
        ref_ffn_bias_2,
    )
    result = flag_gems._transformer_encoder_layer_fwd(
        src,
        embed_dim,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        use_gelu,
        norm_first,
        eps,
        norm_weight_1,
        norm_bias_1,
        norm_weight_2,
        norm_bias_2,
        ffn_weight_1,
        ffn_bias_1,
        ffn_weight_2,
        ffn_bias_2,
    )

    assert result.shape == src.shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype, reduce_dim=seq_len, atol=1e-1)
