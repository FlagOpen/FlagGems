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

SOLVE_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    SOLVE_DTYPES.append(torch.float64)

SOLVE_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]


def _make_solve_inputs(shape, k, dtype, device, *, requires_grad=False):
    torch.manual_seed(0)
    n = shape[-1]
    base = torch.randn(*shape, dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device)
    batch_dims = [1] * (base.ndim - 2)
    if batch_dims:
        eye = eye.view(*batch_dims, n, n)
    matrix = (base @ base.mT + eye * n).detach().requires_grad_(requires_grad)
    rhs = torch.randn(*shape[:-1], k, dtype=dtype, device=device)
    rhs.requires_grad_(requires_grad)
    return matrix, rhs


@pytest.mark.linalg_solve
@pytest.mark.parametrize("shape", SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", SOLVE_DTYPES)
def test_linalg_solve(shape, dtype):
    n, k = shape
    matrix, rhs = _make_solve_inputs((n, n), k, dtype, flag_gems.device)
    matrix_before = matrix.clone()
    rhs_before = rhs.clone()
    ref_matrix = utils.to_reference(matrix, True)
    ref_rhs = utils.to_reference(rhs, True)

    reference = torch.linalg.solve(ref_matrix, ref_rhs)
    with flag_gems.use_gems():
        result = torch.linalg.solve(matrix, rhs)

    utils.gems_assert_close(result, reference, dtype)
    utils.gems_assert_equal(matrix, matrix_before)
    utils.gems_assert_equal(rhs, rhs_before)
    assert result.shape == rhs.shape
    assert result.dtype == dtype
    assert result is not rhs


@pytest.mark.linalg_solve
@pytest.mark.parametrize("batch_shape", [(1,), (3,), (4,), (2, 4)])
@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("k", [1, 4])
@pytest.mark.parametrize("dtype", SOLVE_DTYPES)
def test_linalg_solve_batched(batch_shape, n, k, dtype):
    matrix, rhs = _make_solve_inputs(
        batch_shape + (n, n), k, dtype, flag_gems.device
    )
    ref_matrix = utils.to_reference(matrix, True)
    ref_rhs = utils.to_reference(rhs, True)

    reference = torch.linalg.solve(ref_matrix, ref_rhs)
    with flag_gems.use_gems():
        result = torch.linalg.solve(matrix, rhs)

    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.linalg_solve
@pytest.mark.parametrize("dtype", SOLVE_DTYPES)
def test_linalg_solve_noncontiguous(dtype):
    matrix, _ = _make_solve_inputs((2, 8, 8), 3, dtype, flag_gems.device)
    matrix = matrix.mT
    torch.manual_seed(1)
    rhs_storage = torch.randn((2, 8, 6), dtype=dtype, device=flag_gems.device)
    rhs = rhs_storage[..., ::2]
    assert not matrix.is_contiguous()
    assert not rhs.is_contiguous()
    ref_matrix = utils.to_reference(matrix, True)
    ref_rhs = utils.to_reference(rhs, True)

    reference = torch.linalg.solve(ref_matrix, ref_rhs)
    with flag_gems.use_gems():
        result = torch.linalg.solve(matrix, rhs)

    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.linalg_solve
def test_linalg_solve_empty_batch():
    matrix = torch.empty((0, 4, 4), dtype=torch.float32, device=flag_gems.device)
    rhs = torch.empty((0, 4, 2), dtype=torch.float32, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    ref_rhs = utils.to_reference(rhs, True)

    reference = torch.linalg.solve(ref_matrix, ref_rhs)
    with flag_gems.use_gems():
        result = torch.linalg.solve(matrix, rhs)

    utils.gems_assert_equal(result, reference)
    assert result.shape == rhs.shape


@pytest.mark.linalg_solve
def test_linalg_solve_autograd_is_explicitly_unsupported():
    matrix, rhs = _make_solve_inputs(
        (4, 4), 2, torch.float32, flag_gems.device, requires_grad=True
    )

    with flag_gems.use_gems():
        result = torch.linalg.solve(matrix, rhs)

    assert not result.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(result.square().sum(), (matrix, rhs))


@pytest.mark.linalg_solve
@pytest.mark.parametrize(
    "matrix_shape,rhs_shape",
    [((4, 3), (4, 1)), ((4, 4), (3, 1)), ((4,), (4, 1))],
)
def test_linalg_solve_invalid_shapes(matrix_shape, rhs_shape):
    torch.manual_seed(0)
    matrix = torch.randn(matrix_shape, device=flag_gems.device)
    rhs = torch.randn(rhs_shape, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises((ValueError, RuntimeError)):
        torch.linalg.solve(matrix, rhs)


@pytest.mark.linalg_solve
def test_linalg_solve_right_side_is_rejected_by_current_kernel():
    matrix, rhs = _make_solve_inputs((4, 4), 4, torch.float32, flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(NotImplementedError, match="right=True"):
        torch.linalg.solve(matrix, rhs, left=False)
