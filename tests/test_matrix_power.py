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

MATRIX_POWER_EXPONENTS = [0, 1, 2, 3, 5, 8]

MATRIX_POWER_NEG_EXPONENTS = [-1, -3]

MATRIX_POWER_SHAPES = [
    ((), 2),
    ((), 8),
    ((), 32),
    ((4,), 16),
    ((2, 3), 8),
]

def _make_input(shape, dim, dtype):
    # Build a well-conditioned matrix close to the identity so that repeated
    # squaring keeps element magnitudes bounded. Unit-variance gaussians would
    # blow up geometrically with n and lose all precision in low-precision dtypes.
    eye = torch.eye(dim, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device) * 0.1 + eye

@pytest.mark.matrix_power
@pytest.mark.parametrize("batch_shape, dim", MATRIX_POWER_SHAPES)
@pytest.mark.parametrize("n", MATRIX_POWER_EXPONENTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matrix_power(batch_shape, dim, n, dtype):
    torch.manual_seed(0)
    shape = (*batch_shape, dim, dim)
    res_inp = _make_input(shape, dim, dtype)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.matrix_power(ref_inp, n)
    res_out = flag_gems.matrix_power(res_inp, n)

    # Accumulated error scales with the matrix dimension and the number of products.
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim * max(n, 1))

@pytest.mark.matrix_power
@pytest.mark.parametrize("batch_shape, dim", MATRIX_POWER_SHAPES)
@pytest.mark.parametrize("n", MATRIX_POWER_NEG_EXPONENTS)
# Negative powers require matrix inversion via lu_factor/lu_solve, which torch
# only supports for float32/float64 (bf16/fp16 inverse is not supported by torch).
@pytest.mark.parametrize("dtype", [torch.float32])
def test_matrix_power_negative(batch_shape, dim, n, dtype):
    torch.manual_seed(0)
    shape = (*batch_shape, dim, dim)
    res_inp = _make_input(shape, dim, dtype)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.matrix_power(ref_inp, n)
    res_out = flag_gems.matrix_power(res_inp, n)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim * abs(n))

@pytest.mark.matrix_power
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matrix_power_backward(dtype):
    torch.manual_seed(0)
    inp = _make_input((3, 3), 3, dtype).requires_grad_()
    ref_inp = utils.to_reference(inp, True)
    result = flag_gems.matrix_power(inp, 3)
    reference = torch.matrix_power(ref_inp, 3)
    grad = torch.randn_like(result)
    result_grad = torch.autograd.grad(result, inp, grad)[0]
    reference_grad = torch.autograd.grad(reference, ref_inp, utils.to_reference(grad, True))[0]
    utils.gems_assert_close(result_grad, reference_grad, dtype, reduce_dim=9)
