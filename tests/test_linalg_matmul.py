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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


_CASES = [
    ((5,), (5,)),
    ((5,), (5, 7)),
    ((4, 5), (5,)),
    ((4, 5), (5, 7)),
    ((3, 4, 5), (3, 5, 7)),
    ((1, 4, 5), (3, 5, 7)),
    ((2, 1, 4, 5), (3, 5, 7)),
]
_ZERO_CASES = [
    ((0, 5), (5, 7)),
    ((0, 4, 5), (0, 5, 7)),
    ((5,), (0, 5, 7)),
]


def _compare(left, right, dtype):
    ref_left = utils.to_reference(left, True)
    ref_right = utils.to_reference(right, True)
    expected = torch.linalg.matmul(ref_left, ref_right)
    result = flag_gems.linalg_matmul(left, right)
    assert result.shape == expected.shape
    assert result.dtype == left.dtype
    assert result.device == left.device
    if result.numel() > 0:
        result_storage = result.untyped_storage().data_ptr()
        assert result_storage != left.untyped_storage().data_ptr()
        assert result_storage != right.untyped_storage().data_ptr()
    utils.gems_assert_close(result, expected, dtype, reduce_dim=max(left.shape[-1], 1))


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("left_shape,right_shape", _CASES + _ZERO_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul(left_shape, right_shape, dtype):
    left = torch.randn(left_shape, dtype=dtype, device=flag_gems.device)
    right = torch.randn(right_shape, dtype=dtype, device=flag_gems.device)
    left_before = left.clone()
    right_before = right.clone()

    _compare(left, right, dtype)

    utils.gems_assert_equal(left, left_before)
    utils.gems_assert_equal(right, right_before)


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_noncontiguous(dtype):
    left = torch.randn((4, 9), dtype=dtype, device=flag_gems.device)[:, ::2]
    right = torch.randn((7, 5), dtype=dtype, device=flag_gems.device)[::2, :]
    assert not left.is_contiguous()
    assert not right.is_contiguous()
    _compare(left, right, dtype)


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_grad(dtype):
    left = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device).requires_grad_()
    right = torch.randn((1, 4, 5), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_left = utils.to_reference(left, True).requires_grad_()
    ref_right = utils.to_reference(right, True).requires_grad_()

    result = flag_gems.linalg_matmul(left, right)
    expected = torch.linalg.matmul(ref_left, ref_right)
    grad = torch.randn_like(result)
    ref_grad = utils.to_reference(grad, True)
    result_grads = torch.autograd.grad(result, (left, right), grad)
    expected_grads = torch.autograd.grad(expected, (ref_left, ref_right), ref_grad)

    utils.gems_assert_close(result_grads[0], expected_grads[0], dtype, reduce_dim=5)
    utils.gems_assert_close(result_grads[1], expected_grads[1], dtype, reduce_dim=6)


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_matmul_double_grad(dtype):
    left = torch.randn((3, 4), dtype=dtype, device=flag_gems.device).requires_grad_()
    right = torch.randn((4, 5), dtype=dtype, device=flag_gems.device).requires_grad_()
    ref_left = utils.to_reference(left, True).requires_grad_()
    ref_right = utils.to_reference(right, True).requires_grad_()

    loss = flag_gems.linalg_matmul(left, right).square().sum()
    ref_loss = torch.linalg.matmul(ref_left, ref_right).square().sum()
    grad = torch.autograd.grad(loss, left, create_graph=True)[0]
    ref_grad = torch.autograd.grad(ref_loss, ref_left, create_graph=True)[0]
    double_grad = torch.autograd.grad(grad.sum(), left)[0]
    ref_double_grad = torch.autograd.grad(ref_grad.sum(), ref_left)[0]

    utils.gems_assert_close(double_grad, ref_double_grad, dtype, reduce_dim=5)


@pytest.mark.linalg_matmul
def test_linalg_matmul_rejects_mixed_dtype():
    left = torch.randn((3, 4), dtype=torch.float32, device=flag_gems.device)
    right = torch.randn((4, 5), dtype=torch.float16, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_matmul(left, right)


@pytest.mark.linalg_matmul
@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_linalg_matmul_rejects_unsupported_dtype(dtype):
    left = torch.ones((3, 4), dtype=dtype, device=flag_gems.device)
    right = torch.ones((4, 5), dtype=dtype, device=flag_gems.device)
    with pytest.raises(NotImplementedError):
        flag_gems.linalg_matmul(left, right)


@pytest.mark.linalg_matmul
def test_linalg_matmul_rejects_incompatible_shapes():
    left = torch.randn((3, 4), device=flag_gems.device)
    right = torch.randn((6, 5), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_matmul(left, right)
