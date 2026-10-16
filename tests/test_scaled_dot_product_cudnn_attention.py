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
from .conftest import TO_CPU

CUDNN_ATTENTION = torch.ops.aten._scaled_dot_product_cudnn_attention


@pytest.mark.scaled_dot_product_cudnn_attention
@pytest.mark.skipif(
    flag_gems.device != "cuda",
    reason="aten::_scaled_dot_product_cudnn_attention requires a CUDA/cuDNN reference",
)
@pytest.mark.parametrize("is_causal,use_bias", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_cudnn_attention(is_causal, use_bias, dtype):
    if TO_CPU:
        pytest.skip("cuDNN attention has no CPU reference kernel")
    torch.manual_seed(0)
    shape = (2, 4, 32, 64)
    query = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    key = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    value = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    attn_bias = (
        torch.randn(2, 4, 32, 32, dtype=dtype, device=flag_gems.device) * 0.1
        if use_bias
        else None
    )
    scale = 1.0 / (shape[-1] ** 0.5)

    ref_args = [utils.to_reference(tensor, True) for tensor in (query, key, value)]
    ref_bias = utils.to_reference(attn_bias, True)
    reference = CUDNN_ATTENTION(
        *ref_args,
        attn_bias=ref_bias,
        compute_log_sumexp=True,
        is_causal=is_causal,
        scale=scale,
    )
    with flag_gems.use_gems():
        result = CUDNN_ATTENTION(
            query,
            key,
            value,
            attn_bias=attn_bias,
            compute_log_sumexp=True,
            is_causal=is_causal,
            scale=scale,
        )

    assert len(result) == len(reference) == 9
    utils.gems_assert_close(result[0], reference[0], dtype, atol=5e-2)
    utils.gems_assert_close(result[1], reference[1], torch.float32, atol=5e-2)
    for index in (2, 3, 6, 7, 8):
        assert result[index].shape == reference[index].shape
        assert result[index].dtype == reference[index].dtype
    assert result[4] == reference[4] == shape[2]
    assert result[5] == reference[5] == shape[2]
