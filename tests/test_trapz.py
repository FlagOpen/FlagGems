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


_CASES = [
    pytest.param((128,), -1, 1.0, id="1d-default"),
    pytest.param((64, 128), 0, 0.5, id="2d-first"),
    pytest.param((64, 128), -1, 2.0, id="2d-last"),
    pytest.param((16, 32, 64), 1, 0.5, id="3d-middle"),
]


def _assert_trapz(inp, dtype, dx, dim):
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.trapezoid(ref_inp, dx=dx, dim=dim)
    result = flag_gems.trapz(inp, dx=dx, dim=dim)

    assert result.shape == ref_out.shape
    assert result.dtype == ref_out.dtype
    utils.gems_assert_close(
        result, ref_out, dtype, reduce_dim=max(inp.shape[dim], 1)
    )


@pytest.mark.trapz
@pytest.mark.parametrize("shape,dim,dx", _CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz(shape, dim, dx, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    _assert_trapz(inp, dtype, dx, dim)


@pytest.mark.trapz
@pytest.mark.parametrize("length", [1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_short_reduction(length, dtype):
    torch.manual_seed(0)
    inp = torch.randn(length, dtype=dtype, device=flag_gems.device)
    _assert_trapz(inp, dtype, 1.0, 0)


@pytest.mark.trapz
@pytest.mark.parametrize("shape", [(1024, 1024), (512, 2048)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_large_reduction(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    _assert_trapz(inp, dtype, 0.01, -1)


@pytest.mark.trapz
@pytest.mark.parametrize("shape", [(2, 0, 3), (3, 1, 4)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_empty_or_single_length(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    _assert_trapz(inp, dtype, 1.0, 1)


@pytest.mark.trapz
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_noncontiguous(dim, dtype):
    torch.manual_seed(0)
    inp = torch.randn((8, 6, 4), dtype=dtype, device=flag_gems.device).transpose(0, 2)
    assert not inp.is_contiguous()
    _assert_trapz(inp, dtype, 1.0, dim)


@pytest.mark.trapz
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_trapz_integer_promotion(dtype):
    torch.manual_seed(0)
    inp = torch.randint(-8, 8, (3, 4), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    ref_out = torch.trapezoid(ref_inp, dx=0.5, dim=-1)
    result = flag_gems.trapz(inp, dx=0.5, dim=-1)

    assert result.dtype == ref_out.dtype == torch.float32
    utils.gems_assert_close(result, ref_out, torch.float32, reduce_dim=4)


@pytest.mark.trapz
@pytest.mark.skipif(not utils.fp64_is_supported, reason="fp64 not supported")
@pytest.mark.parametrize("shape,dim", [((64, 128), -1), ((16, 32, 64), 0)])
@pytest.mark.parametrize("dx", [0.5, 0.1])
def test_trapz_fp64(shape, dim, dx):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)
    _assert_trapz(inp, torch.float64, dx, dim)


@pytest.mark.trapz
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_dispatcher(dim, dtype):
    torch.manual_seed(0)
    inp = torch.randn((4, 7), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.trapezoid(ref_inp, dx=1.5, dim=dim)

    with flag_gems.use_gems():
        result = torch.trapezoid(inp, dx=1.5, dim=dim)

    utils.gems_assert_close(result, ref_out, dtype, reduce_dim=inp.shape[dim])


@pytest.mark.trapz
@pytest.mark.parametrize("dim", [0, 2, -3])
def test_trapz_rejects_invalid_dim(dim):
    shape = () if dim == 0 else (2, 3)
    inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(IndexError):
        flag_gems.trapz(inp, dx=1.0, dim=dim)


@pytest.mark.trapz
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_trapz_rejects_complex(dtype):
    torch.manual_seed(0)
    inp = torch.randn((3, 4), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.trapz(inp, dx=1.0, dim=-1)


@pytest.mark.trapz
def test_trapz_rejects_bool():
    inp = torch.tensor([[True, False], [False, True]], device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.trapz(inp, dx=1.0, dim=-1)


@pytest.mark.trapz
@pytest.mark.parametrize("shape,dim", [((16, 32), -1), ((4, 8, 16), 0)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_backward(shape, dim, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.trapezoid(ref_inp, dx=2.0, dim=dim)
    result = flag_gems.trapz(inp, dx=2.0, dim=dim)
    out_grad = torch.randn_like(result)
    ref_grad = utils.to_reference(out_grad, True)

    expected = torch.autograd.grad(ref_out, ref_inp, ref_grad)[0]
    actual = torch.autograd.grad(result, inp, out_grad)[0]

    utils.gems_assert_close(actual, expected, dtype)
