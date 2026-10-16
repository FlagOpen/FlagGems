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


_SHAPES = [*utils.POINTWISE_SHAPES, (0,), (2, 0, 3)]


def _assert_frexp(inp, dtype):
    ref_inp = utils.to_reference(inp, True)
    ref_mantissa, ref_exponent = torch.frexp(ref_inp)

    with flag_gems.use_gems():
        mantissa, exponent = torch.frexp(inp)

    assert mantissa.shape == exponent.shape == inp.shape
    assert mantissa.dtype == dtype
    assert exponent.dtype == ref_exponent.dtype
    utils.gems_assert_close(mantissa, ref_mantissa, dtype)
    utils.gems_assert_equal(exponent, ref_exponent)


@pytest.mark.frexp
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_frexp(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    _assert_frexp(inp, dtype)


@pytest.mark.frexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_frexp_noncontiguous(dtype):
    torch.manual_seed(0)
    inp = torch.randn((4, 7, 9), dtype=dtype, device=flag_gems.device).transpose(0, 2)
    assert not inp.is_contiguous()
    _assert_frexp(inp, dtype)


@pytest.mark.frexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_frexp_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -2.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    _assert_frexp(inp, dtype)


@pytest.mark.frexp
def test_frexp_rejects_integer_input():
    inp = torch.arange(8, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.frexp(utils.to_reference(inp, False))
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.frexp(inp)
