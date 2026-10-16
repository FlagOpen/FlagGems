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

FORWARD = torch.ops.aten._scaled_dot_product_cudnn_attention
BACKWARD = torch.ops.aten._scaled_dot_product_cudnn_attention_backward


@pytest.mark.scaled_dot_product_cudnn_attention_backward
@pytest.mark.skipif(
    flag_gems.device != "cuda",
    reason="aten::_scaled_dot_product_cudnn_attention_backward requires CUDA/cuDNN",
)
@pytest.mark.parametrize("is_causal,use_bias", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_cudnn_attention_backward(is_causal, use_bias, dtype):
    if TO_CPU:
        pytest.skip("cuDNN attention backward has no CPU reference kernel")
    torch.manual_seed(0)
    shape = (1, 2, 32, 64)
    query = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    key = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    value = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    attn_bias = (
        torch.randn(1, 2, 32, 32, dtype=dtype, device=flag_gems.device) * 0.1
        if use_bias
        else None
    )
    scale = 1.0 / (shape[-1] ** 0.5)

    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_value = utils.to_reference(value, True)
    ref_grad = utils.to_reference(grad_out, True)
    ref_bias = utils.to_reference(attn_bias, True)
    forward = FORWARD(
        ref_query, ref_key, ref_value, ref_bias, True, 0.0, is_causal, False, scale=scale
    )
    ref_grads = BACKWARD(
        ref_grad,
        ref_query,
        ref_key,
        ref_value,
        forward[0],
        forward[1],
        forward[6],
        forward[7],
        ref_bias,
        forward[2],
        forward[3],
        forward[4],
        forward[5],
        0.0,
        is_causal,
        scale=scale,
    )

    target_forward = FORWARD(
        query, key, value, attn_bias, True, 0.0, is_causal, False, scale=scale
    )
    with flag_gems.use_gems():
        grads = BACKWARD(
            grad_out,
            query,
            key,
            value,
            target_forward[0],
            target_forward[1],
            target_forward[6],
            target_forward[7],
            attn_bias,
            target_forward[2],
            target_forward[3],
            target_forward[4],
            target_forward[5],
            0.0,
            is_causal,
            scale=scale,
        )

    assert len(grads) == len(ref_grads) == 3
    for result, reference in zip(grads, ref_grads):
        assert result.shape == reference.shape
        utils.gems_assert_close(result, reference, dtype, atol=5e-2, equal_nan=True)
