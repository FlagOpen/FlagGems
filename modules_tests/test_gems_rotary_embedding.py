import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.modules import GemsDeepseekYarnRoPE
from flag_gems.testing import assert_close

from .module_test_util import has_c_extension, has_vllm, is_torch_version_ge, init_seed

device = flag_gems.device


@pytest.mark.parametrize("shape", [(2, 128), (4, 256), (8, 64)])  # (seq_len, head_dim)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rotary_interleaved", [False, True])
def test_gems_deepseek_rope(shape, dtype, rotary_interleaved):

    init_seed(42)

    seq_len, head_dim = shape
    batch_size = 2
    num_heads = 4

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    key = torch.randn_like(query)

    head_size = 64
    rotary_dim = 64
    max_position = 2048
    original_max_position = 4096
    base = 10000
    scaling_factor = 1.0

    if has_vllm():
        from vllm.model_executor.layers.rotary_embedding import DeepseekScalingRotaryEmbedding
        is_neox_style = not rotary_interleaved

        vllm_rotary_emb = DeepseekScalingRotaryEmbedding(
            head_size, rotary_dim, original_max_position, base,
            is_neox_style, scaling_factor, dtype)
        
        vllm_query = query.clone()
        vllm_key = key.clone()
        vllm_query, vllm_key = vllm_rotary_emb(vllm_query, vllm_key)

        assert_close(out_test, vllm_query, dtype, reduce_dim=norm_shape)
        assert_close(new_residual, vllm_key, dtype)
    else:
        pytest.skip("Skipping vLLM DeepseekScalingRotaryEmbedding comparison: vLLM not installed")

