import pytest
import torch

import flag_gems
from flag_gems.modules import GemsDeepseekYarnRoPE
from flag_gems.testing import assert_close

from .module_test_util import has_vllm, init_seed

device = flag_gems.device

# @pytest.mark.parametrize("batch_size", [8, 32])
# @pytest.mark.parametrize("max_seq_len", [2048, 4096])
# @pytest.mark.parametrize("q_heads, k_heads", [(8, 1), (8, 8)])
# @pytest.mark.parametrize("head_dim", [64, 128, 256])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("rotary_interleaved", [False])
# def test_gems_rope(batch_size,
#                     max_seq_len,
#                     q_heads,
#                     k_heads,
#                     head_dim,
#                     dtype,
#                     rotary_interleaved):
#     init_seed(42)
#     base = 10000
#     rotary_dim = head_dim

#     seq_len = torch.randint(1, (max_seq_len // 2), (1,)).item()

#     query = torch.randn(
#         (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=device)
#     key = torch.randn(
#         (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=device)

#     position_ids = torch.randint(
#         0, max_seq_len, (batch_size, seq_len), device=device
#     )

#     if has_vllm():
#         is_neox_style = not rotary_interleaved
#         head_size = rotary_dim
#         original_max_position = max_seq_len

#         from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
#         vllm_rotary_emb = RotaryEmbedding(
#             head_size,
#             rotary_dim,
#             original_max_position,
#             base,
#             is_neox_style,
#             dtype)

#         # try to test the accuracy with or without gems patches
#         flag_gems.apply_gems_patches_to_vllm(verbose=True)
#         vllm_rotary_emb_path =  RotaryEmbedding(
#             head_size,
#             rotary_dim,
#             original_max_position,
#             base,
#             is_neox_style,
#             dtype)

#         vllm_query = query.clone()
#         vllm_key = key.clone()
#         query, key = vllm_rotary_emb_path(position_ids, query, key)
#         vllm_query, vllm_key = vllm_rotary_emb(position_ids, vllm_query, vllm_key)

#         assert_close(query, vllm_query, dtype)
#         assert_close(key, vllm_key, dtype)

#     else:
#         pytest.skip("Skipping vLLM RotaryEmbedding comparison: vLLM not installed")


@pytest.mark.parametrize("batch_size", [8, 32])
@pytest.mark.parametrize("max_seq_len", [2048, 4096])
@pytest.mark.parametrize("q_heads, k_heads", [(8, 1), (8, 8)])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rotary_interleaved", [True, False])
def test_gems_deepseek_rope(
    batch_size, max_seq_len, q_heads, k_heads, head_dim, dtype, rotary_interleaved
):
    init_seed(42)
    base = 10000

    rope_scaling = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    }

    scaling_type = rope_scaling["type"]
    scaling_factor = rope_scaling["factor"]
    assert scaling_type == "yarn", "Only 'yarn' scaling type is supported in this test."

    kwargs = {
        k: v
        for k, v in rope_scaling.items()
        if k
        in (
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
        )
    }

    seq_len = torch.randint(1, (max_seq_len // 2), (1,)).item()

    query = torch.randn(
        (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=device
    )
    key = torch.randn(
        (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=device
    )

    position_ids = torch.randint(0, max_seq_len, (batch_size, seq_len), device=device)

    rotary_dim = head_dim
    max_position_embeddings = rope_scaling["original_max_position_embeddings"] // 2

    gems_rope = GemsDeepseekYarnRoPE(
        dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        device=device,
        scaling_factor=scaling_factor,
        **kwargs,
        forward_only=True,
    ).to(dtype=dtype, device=device)

    out_test, new_residual = gems_rope(
        query, key, position_ids, rotary_interleaved, seq_len, True
    )

    if has_vllm():
        from vllm.model_executor.layers.rotary_embedding import (
            DeepseekScalingRotaryEmbedding,
        )

        is_neox_style = not rotary_interleaved
        head_size = rotary_dim

        original_max_position = rope_scaling["original_max_position_embeddings"]

        extra_kwargs = {
            k: v
            for k, v in rope_scaling.items()
            if k
            in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
        }

        vllm_rotary_emb = DeepseekScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            original_max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            **extra_kwargs,
        )

        # try to test the accuracy with or without gems patches
        flag_gems.apply_gems_patches_to_vllm(verbose=True)
        vllm_rotary_emb_path = DeepseekScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            original_max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            **extra_kwargs,
        )

        vllm_query = query.clone()
        vllm_key = key.clone()
        query, key = vllm_rotary_emb_path(position_ids, query, key)
        vllm_query, vllm_key = vllm_rotary_emb(position_ids, vllm_query, vllm_key)

        assert_close(query, vllm_query, dtype)
        assert_close(key, vllm_key, dtype)
    else:
        pytest.skip(
            "Skipping vLLM DeepseekScalingRotaryEmbedding comparison: vLLM not installed"
        )
