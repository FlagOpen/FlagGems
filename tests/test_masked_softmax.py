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


def _assert_masked_softmax(input, mask, dim, mask_type, dtype):
    reference = torch.ops.aten._masked_softmax(
        utils.to_reference(input, True),
        utils.to_reference(mask, False),
        dim,
        mask_type,
    )
    result = flag_gems._masked_softmax(input, mask, dim, mask_type)
    assert result.shape == input.shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype, equal_nan=True)


@pytest.mark.masked_softmax
@pytest.mark.parametrize("shape", [(128, 256), (8, 16, 64), (64, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_masked_softmax_elementwise(shape, dtype, dim):
    torch.manual_seed(0)
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    _assert_masked_softmax(input, mask, dim, 2, dtype)


@pytest.mark.masked_softmax
@pytest.mark.parametrize("shape", [(4, 8, 32, 32), (2, 4, 64, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("mask_type", [0, 1])
def test_masked_softmax_attention_masks(shape, dtype, mask_type):
    torch.manual_seed(0)
    batch, _, _, seq_len = shape
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask_shape = (seq_len, seq_len) if mask_type == 0 else (batch, seq_len)
    mask = torch.randint(
        0, 2, mask_shape, dtype=torch.bool, device=flag_gems.device
    )
    if mask_type == 1:
        mask[:, 0] = False
    else:
        mask[..., 0] = False
    _assert_masked_softmax(input, mask, -1, mask_type, dtype)


@pytest.mark.masked_softmax
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_masked_softmax_all_masked_row(dtype):
    torch.manual_seed(0)
    input = torch.randn((4, 16), dtype=dtype, device=flag_gems.device)
    mask = torch.zeros_like(input, dtype=torch.bool)
    mask[0] = True
    _assert_masked_softmax(input, mask, -1, 2, dtype)
