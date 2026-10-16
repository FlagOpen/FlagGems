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

@pytest.mark.matrix_power_out
@pytest.mark.parametrize("batch_shape, dim", MATRIX_POWER_SHAPES)
@pytest.mark.parametrize("n", MATRIX_POWER_EXPONENTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matrix_power_out(batch_shape, dim, n, dtype):
    torch.manual_seed(0)
    shape = (*batch_shape, dim, dim)
    res_inp = _make_input(shape, dim, dtype)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.matrix_power(ref_inp, n, out=ref_out)

    res_out = torch.empty_like(res_inp)
    out_pointer = res_out.data_ptr()
    result = flag_gems.matrix_power_out(res_inp, n, out=res_out)

    assert result is res_out
    assert res_out.data_ptr() == out_pointer
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim * max(n, 1))


@pytest.mark.matrix_power_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matrix_power_out_resizes_empty(dtype):
    torch.manual_seed(0)
    inp = _make_input((4, 4), 4, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.empty(0, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.matrix_power(ref_inp, 2, out=ref_out)
    out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    result = flag_gems.matrix_power_out(inp, 2, out=out)
    assert result is out
    assert tuple(out.shape) == (4, 4)
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=8)


@pytest.mark.matrix_power_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matrix_power_out_noncontiguous(dtype):
    torch.manual_seed(0)
    inp = _make_input((4, 4), 4, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.empty((4, 4), dtype=ref_inp.dtype, device=ref_inp.device).t()
    torch.matrix_power(ref_inp, 3, out=ref_out)
    out = torch.empty((4, 4), dtype=dtype, device=flag_gems.device).t()
    out_pointer = out.data_ptr()

    result = flag_gems.matrix_power_out(inp, 3, out=out)

    assert result is out
    assert out.data_ptr() == out_pointer
    assert not out.is_contiguous()
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=12)


@pytest.mark.matrix_power_out
def test_matrix_power_out_dtype_conversion():
    torch.manual_seed(0)
    inp = _make_input((4, 4), 4, torch.float32)
    ref_inp = utils.to_reference(inp, True).to(torch.float32)
    ref_out = torch.empty((4, 4), dtype=torch.float64, device=ref_inp.device)
    torch.matrix_power(ref_inp, 3, out=ref_out)
    out = torch.empty((4, 4), dtype=torch.float64, device=flag_gems.device)
    out_pointer = out.data_ptr()

    result = flag_gems.matrix_power_out(inp, 3, out=out)

    assert result is out
    assert out.data_ptr() == out_pointer
    assert out.dtype == torch.float64
    utils.gems_assert_close(out, ref_out, torch.float32, reduce_dim=12)


@pytest.mark.matrix_power_out
def test_matrix_power_out_rejects_integer_out():
    torch.manual_seed(0)
    inp = _make_input((4, 4), 4, torch.float32)
    ref_inp = utils.to_reference(inp, True).to(torch.float32)
    ref_out = torch.empty((4, 4), dtype=torch.int64, device=ref_inp.device)
    with pytest.raises(RuntimeError):
        torch.matrix_power(ref_inp, 3, out=ref_out)

    out = torch.empty((4, 4), dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.matrix_power_out(inp, 3, out=out)
