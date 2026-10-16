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

FORWARD = torch.ops.aten._scaled_dot_product_efficient_attention
BACKWARD = torch.ops.aten._scaled_dot_product_efficient_attention_backward
CASES = [(False, (True, True, True, False)), (True, (True, True, True, False)), (False, (True, False, True, False))]


@pytest.mark.scaled_dot_product_efficient_attention_backward
@pytest.mark.parametrize("is_causal,grad_input_mask", CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_efficient_attention_backward(is_causal, grad_input_mask, dtype):
    torch.manual_seed(0)
    shape = (1, 2, 32, 64)
    query = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    key = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    value = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scale = 1.0 / (shape[-1] ** 0.5)

    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_value = utils.to_reference(value, True)
    ref_grad = utils.to_reference(grad_out, True)
    ref_forward = FORWARD(ref_query, ref_key, ref_value, None, True, 0.0, is_causal, scale=scale)
    ref_grads = BACKWARD(
        ref_grad, ref_query, ref_key, ref_value, None, ref_forward[0], ref_forward[1],
        ref_forward[2], ref_forward[3], 0.0, list(grad_input_mask), is_causal, scale=scale,
    )

    target_forward = FORWARD(query, key, value, None, True, 0.0, is_causal, scale=scale)
    with flag_gems.use_gems():
        grads = BACKWARD(
            grad_out, query, key, value, None, target_forward[0], target_forward[1],
            target_forward[2], target_forward[3], 0.0, list(grad_input_mask), is_causal,
            scale=scale,
        )

    assert len(grads) == len(ref_grads) == 4
    for needed, result, reference in zip(grad_input_mask, grads, ref_grads):
        if needed:
            assert result.shape == reference.shape
            utils.gems_assert_close(result, reference, dtype, atol=5e-2, equal_nan=True)
        else:
            utils.gems_assert_equal(result, torch.zeros_like(result))
