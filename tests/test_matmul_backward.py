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

MATMUL_SHAPES = [
    ((16, 32), (32, 24)),
    ((64, 128), (128, 64)),
    ((4, 16, 32), (4, 32, 24)),
    ((8, 32, 64), (8, 64, 16)),
    ((2, 3, 16, 32), (2, 3, 32, 24)),
    ((32,), (32,)),
    ((32,), (32, 24)),
    ((16, 32), (32,)),
    ((4, 16, 32), (32, 24)),
    ((16, 32), (4, 32, 24)),
    ((5, 1, 16, 32), (1, 3, 32, 24)),
]
MASKS = [(True, True), (True, False), (False, True), (False, False)]


def _reference_grads(self_tensor, other_tensor, grad, mask):
    self_tensor = self_tensor.detach().requires_grad_(True)
    other_tensor = other_tensor.detach().requires_grad_(True)
    result = torch.matmul(self_tensor, other_tensor)
    requested = []
    if mask[0]:
        requested.append(self_tensor)
    if mask[1]:
        requested.append(other_tensor)
    if not requested:
        return None, None
    gradients = torch.autograd.grad(result, requested, grad)
    offset = 0
    grad_self = None
    grad_other = None
    if mask[0]:
        grad_self = gradients[offset]
        offset += 1
    if mask[1]:
        grad_other = gradients[offset]
    return grad_self, grad_other


def _assert_optional_close(actual, expected, dtype, reduce_dim):
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert actual.shape == expected.shape
        assert actual.dtype == dtype
        utils.gems_assert_close(actual, expected, dtype, reduce_dim=reduce_dim)


@pytest.mark.matmul_backward
@pytest.mark.parametrize("self_shape,other_shape", MATMUL_SHAPES)
@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmul_backward(self_shape, other_shape, mask, dtype):
    torch.manual_seed(0)
    self_tensor = torch.randn(
        self_shape, dtype=dtype, device=flag_gems.device
    )
    other_tensor = torch.randn(
        other_shape, dtype=dtype, device=flag_gems.device
    )
    grad = torch.randn_like(torch.matmul(self_tensor, other_tensor))
    saved_self = self_tensor.clone()
    saved_other = other_tensor.clone()
    saved_grad = grad.clone()
    ref_self = utils.to_reference(self_tensor, True)
    ref_other = utils.to_reference(other_tensor, True)
    ref_grad = utils.to_reference(grad, True)
    ref_grad_self, ref_grad_other = _reference_grads(
        ref_self, ref_other, ref_grad, mask
    )

    grad_self, grad_other = flag_gems.matmul_backward(
        grad, self_tensor, other_tensor, mask
    )

    reduce_dim = self_tensor.shape[-1]
    _assert_optional_close(grad_self, ref_grad_self, dtype, reduce_dim)
    _assert_optional_close(grad_other, ref_grad_other, dtype, reduce_dim)
    utils.gems_assert_equal(self_tensor, saved_self)
    utils.gems_assert_equal(other_tensor, saved_other)
    utils.gems_assert_equal(grad, saved_grad)


@pytest.mark.matmul_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmul_backward_noncontiguous_broadcast(dtype):
    torch.manual_seed(1)
    self_tensor = torch.randn(
        (2, 1, 32, 16), dtype=dtype, device=flag_gems.device
    ).transpose(-2, -1)
    other_tensor = torch.randn(
        (1, 3, 24, 32), dtype=dtype, device=flag_gems.device
    ).transpose(-2, -1)
    assert not self_tensor.is_contiguous()
    assert not other_tensor.is_contiguous()
    grad = torch.randn_like(torch.matmul(self_tensor, other_tensor))
    ref_self = utils.to_reference(self_tensor, True)
    ref_other = utils.to_reference(other_tensor, True)
    ref_grad = utils.to_reference(grad, True)
    reference = _reference_grads(ref_self, ref_other, ref_grad, (True, True))

    result = flag_gems.matmul_backward(
        grad, self_tensor, other_tensor, (True, True)
    )

    utils.gems_assert_close(result[0], reference[0], dtype, reduce_dim=32)
    utils.gems_assert_close(result[1], reference[1], dtype, reduce_dim=32)


@pytest.mark.matmul_backward
def test_matmul_backward_rejects_short_mask():
    torch.manual_seed(0)
    self_tensor = torch.randn((4, 8), device=flag_gems.device)
    other_tensor = torch.randn((8, 3), device=flag_gems.device)
    grad = torch.randn((4, 3), device=flag_gems.device)

    with pytest.raises((IndexError, RuntimeError)):
        flag_gems.matmul_backward(grad, self_tensor, other_tensor, [True])


@pytest.mark.matmul_backward
def test_matmul_backward_rejects_incompatible_shapes():
    torch.manual_seed(0)
    self_tensor = torch.randn((4, 8), device=flag_gems.device)
    other_tensor = torch.randn((7, 3), device=flag_gems.device)
    grad = torch.randn((4, 3), device=flag_gems.device)

    with pytest.raises((RuntimeError, AssertionError)):
        flag_gems.matmul_backward(
            grad, self_tensor, other_tensor, (True, True)
        )
