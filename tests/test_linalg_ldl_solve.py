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


LDL_SOLVE_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (32, 7),
]
LDL_SOLVE_DTYPES = [torch.float32, torch.float64, torch.complex64]
if utils.fp64_is_supported:
    LDL_SOLVE_DTYPES.append(torch.complex128)
HERMITIAN_DTYPES = [torch.complex64]
if utils.fp64_is_supported:
    HERMITIAN_DTYPES.append(torch.complex128)


def _make_system(batch_shape, n, k, dtype, device, *, hermitian=False):
    torch.manual_seed(0)
    matrix = torch.randn(*batch_shape, n, n, dtype=dtype, device=device)
    transpose = matrix.mH if hermitian else matrix.mT
    matrix = matrix @ transpose + n * torch.eye(n, dtype=dtype, device=device)
    rhs = torch.randn(*batch_shape, n, k, dtype=dtype, device=device)
    return matrix, rhs


def _factor_and_solve(matrix, rhs, *, hermitian=False):
    ld, pivots, info = torch.linalg.ldl_factor_ex(matrix, hermitian=hermitian)
    assert info.eq(0).all()
    return torch.linalg.ldl_solve(ld, pivots, rhs, hermitian=hermitian)


def _target_factor_and_solve(matrix, rhs, *, hermitian=False):
    ld, pivots, info = torch.linalg.ldl_factor_ex(matrix, hermitian=hermitian)
    assert info.eq(0).all()
    return flag_gems.linalg_ldl_solve(ld, pivots, rhs, hermitian=hermitian)


def _assert_solve_close(result, reference, dtype):
    if dtype == torch.complex128:
        result = utils.to_cpu(result, reference)
        torch.testing.assert_close(result, reference, atol=1e-7, rtol=1e-7)
    else:
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.linalg_ldl_solve
@pytest.mark.parametrize("shape", LDL_SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", LDL_SOLVE_DTYPES)
def test_linalg_ldl_solve(shape, dtype):
    n, k = shape
    matrix, rhs = _make_system((), n, k, dtype, flag_gems.device)
    rhs_before = rhs.clone()
    upcast = dtype in (torch.float32, torch.complex64)
    ref_matrix = utils.to_reference(matrix, upcast)
    ref_rhs = utils.to_reference(rhs, upcast)

    reference = _factor_and_solve(ref_matrix, ref_rhs)
    result = _target_factor_and_solve(matrix, rhs)

    _assert_solve_close(result, reference, dtype)
    utils.gems_assert_equal(rhs, rhs_before)
    assert result is not rhs


@pytest.mark.linalg_ldl_solve
@pytest.mark.parametrize("shape", LDL_SOLVE_SHAPES)
@pytest.mark.parametrize("dtype", LDL_SOLVE_DTYPES)
def test_linalg_ldl_solve_batched(shape, dtype):
    n, k = shape
    matrix, rhs = _make_system((4,), n, k, dtype, flag_gems.device)
    upcast = dtype in (torch.float32, torch.complex64)
    ref_matrix = utils.to_reference(matrix, upcast)
    ref_rhs = utils.to_reference(rhs, upcast)

    reference = _factor_and_solve(ref_matrix, ref_rhs)
    result = _target_factor_and_solve(matrix, rhs)

    _assert_solve_close(result, reference, dtype)


@pytest.mark.linalg_ldl_solve
@pytest.mark.parametrize("dtype", HERMITIAN_DTYPES)
def test_linalg_ldl_solve_hermitian(dtype):
    matrix, rhs = _make_system(
        (), 8, 3, dtype, flag_gems.device, hermitian=True
    )
    upcast = dtype == torch.complex64
    ref_matrix = utils.to_reference(matrix, upcast)
    ref_rhs = utils.to_reference(rhs, upcast)

    reference = _factor_and_solve(ref_matrix, ref_rhs, hermitian=True)
    result = _target_factor_and_solve(matrix, rhs, hermitian=True)

    _assert_solve_close(result, reference, dtype)


@pytest.mark.linalg_ldl_solve
def test_linalg_ldl_solve_noncontiguous_rhs():
    matrix, rhs_base = _make_system((), 8, 3, torch.float32, flag_gems.device)
    rhs = rhs_base.transpose(0, 1).contiguous().transpose(0, 1)
    assert not rhs.is_contiguous()
    ref_matrix = utils.to_reference(matrix, True)
    ref_rhs = utils.to_reference(rhs, True)

    reference = _factor_and_solve(ref_matrix, ref_rhs)
    result = _target_factor_and_solve(matrix, rhs)

    utils.gems_assert_close(result, reference, torch.float32)


@pytest.mark.linalg_ldl_solve
def test_linalg_ldl_solve_empty_rhs_returns_copy():
    ld = torch.empty((0, 0), dtype=torch.float32, device=flag_gems.device)
    pivots = torch.empty((0,), dtype=torch.int32, device=flag_gems.device)
    rhs = torch.empty((0, 2), dtype=torch.float32, device=flag_gems.device)

    result = flag_gems.linalg_ldl_solve(ld, pivots, rhs)

    assert result is not rhs
    assert result.shape == rhs.shape
    assert result.dtype == rhs.dtype
    utils.gems_assert_equal(result, rhs)


@pytest.mark.linalg_ldl_solve
@pytest.mark.parametrize(
    "ld_shape,pivots_shape,rhs_shape,ld_dtype,rhs_dtype,error_type,message",
    [
        ((4,), (4,), (4, 1), torch.float32, torch.float32, ValueError, "at least 2D"),
        ((4, 3), (4,), (4, 1), torch.float32, torch.float32, ValueError, "square"),
        ((4, 4), (4,), (3, 1), torch.float32, torch.float32, ValueError, "same n"),
        ((4, 4), (3,), (4, 1), torch.float32, torch.float32, ValueError, "pivots"),
        ((2, 4, 4), (2, 4), (3, 4, 1), torch.float32, torch.float32, ValueError, "batch dimensions"),
        ((4, 4), (4,), (4, 1), torch.float32, torch.float64, TypeError, "same dtype"),
        ((4, 4), (4,), (4, 1), torch.int32, torch.int32, TypeError, "supports only"),
    ],
)
def test_linalg_ldl_solve_rejects_invalid_inputs(
    ld_shape,
    pivots_shape,
    rhs_shape,
    ld_dtype,
    rhs_dtype,
    error_type,
    message,
):
    ld = torch.ones(ld_shape, dtype=ld_dtype, device=flag_gems.device)
    pivots = torch.ones(pivots_shape, dtype=torch.int32, device=flag_gems.device)
    rhs = torch.ones(rhs_shape, dtype=rhs_dtype, device=flag_gems.device)

    with pytest.raises(error_type, match=message):
        flag_gems.linalg_ldl_solve(ld, pivots, rhs)
