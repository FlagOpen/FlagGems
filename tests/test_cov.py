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


_COV_SHAPES = [(1, 8), (2, 4), (4, 16), (8, 64), (32, 128)]


def _assert_cov(inp, result, reference, dtype):
    expected_shape = () if inp.ndim == 1 else (inp.shape[0], inp.shape[0])
    assert tuple(result.shape) == expected_shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype, reduce_dim=inp.shape[-1])


@pytest.mark.cov
@pytest.mark.parametrize("shape", _COV_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    before = inp.clone()
    ref_inp = utils.to_reference(inp, True)
    reference = torch.cov(ref_inp).to(dtype)

    with flag_gems.use_gems():
        result = torch.cov(inp)

    _assert_cov(inp, result, reference, dtype)
    utils.gems_assert_equal(inp, before)


@pytest.mark.cov
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov_one_dimensional(dtype):
    torch.manual_seed(0)
    inp = torch.randn((64,), dtype=dtype, device=flag_gems.device)
    reference = torch.cov(utils.to_reference(inp, True)).to(dtype)

    with flag_gems.use_gems():
        result = torch.cov(inp)

    _assert_cov(inp, result, reference, dtype)


@pytest.mark.cov
@pytest.mark.parametrize("correction", [0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov_correction(correction, dtype):
    torch.manual_seed(0)
    inp = torch.randn((8, 64), dtype=dtype, device=flag_gems.device)
    reference = torch.cov(
        utils.to_reference(inp, True), correction=correction
    ).to(dtype)

    with flag_gems.use_gems():
        result = torch.cov(inp, correction=correction)

    _assert_cov(inp, result, reference, dtype)


@pytest.mark.cov
@pytest.mark.parametrize("weight_mode", ["frequency", "analytic", "both"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov_weights(weight_mode, dtype):
    torch.manual_seed(0)
    observations = 128
    inp = torch.randn((8, observations), dtype=dtype, device=flag_gems.device)
    fweights = (
        torch.randint(1, 5, (observations,), device=flag_gems.device)
        if weight_mode in ("frequency", "both")
        else None
    )
    aweights = (
        torch.rand((observations,), dtype=dtype, device=flag_gems.device) + 0.1
        if weight_mode in ("analytic", "both")
        else None
    )
    ref_inp = utils.to_reference(inp, True)
    ref_fweights = (
        utils.to_reference(fweights, False) if fweights is not None else None
    )
    ref_aweights = (
        utils.to_reference(aweights, True) if aweights is not None else None
    )
    reference = torch.cov(
        ref_inp, fweights=ref_fweights, aweights=ref_aweights
    ).to(dtype)

    with flag_gems.use_gems():
        result = torch.cov(inp, fweights=fweights, aweights=aweights)

    _assert_cov(inp, result, reference, dtype)


@pytest.mark.cov
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cov_noncontiguous(dtype):
    torch.manual_seed(0)
    base = torch.randn((64, 256), dtype=dtype, device=flag_gems.device)
    inp = base[::2, ::2]
    assert not inp.is_contiguous()
    reference = torch.cov(utils.to_reference(inp, True)).to(dtype)

    with flag_gems.use_gems():
        result = torch.cov(inp)

    _assert_cov(inp, result, reference, dtype)


@pytest.mark.cov
def test_cov_rejects_invalid_frequency_weights():
    inp = torch.randn((4, 16), device=flag_gems.device)
    wrong_length = torch.ones((15,), dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.cov(inp, fweights=wrong_length)
