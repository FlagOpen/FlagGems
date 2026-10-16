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


_NDTRI_DTYPES = [torch.float32] + (
    [torch.float64] if utils.fp64_is_supported else []
)
_NDTRI_UNSUPPORTED_DTYPES = [torch.float16, torch.bfloat16]


def _probability_input(shape, dtype):
    torch.manual_seed(0)
    return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
        0.01, 0.99
    )


def _assert_ndtri_dispatch(caplog):
    expected = f"{utils.gems_log_prefix(flag_gems.special_ndtri)} SPECIAL_NDTRI"
    assert expected in caplog.text


@pytest.mark.special_ndtri
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _NDTRI_DTYPES)
def test_special_ndtri(shape, dtype, caplog):
    inp = _probability_input(shape, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.ops.aten.special_ndtri(ref_inp)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_ndtri"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_ndtri(inp)

    _assert_ndtri_dispatch(caplog)
    assert res_out.shape == inp.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_ndtri
@pytest.mark.parametrize("dtype", _NDTRI_DTYPES)
def test_special_ndtri_edge_values(dtype, caplog):
    inp = torch.tensor(
        [0.0, 1.0, 0.5, -0.1, 1.1, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.ops.aten.special_ndtri(ref_inp).to(dtype=dtype)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_ndtri"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_ndtri(inp)

    _assert_ndtri_dispatch(caplog)
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.special_ndtri
@pytest.mark.parametrize("dtype", _NDTRI_DTYPES)
def test_special_ndtri_noncontiguous(dtype, caplog):
    inp = _probability_input((16, 32), dtype).t()
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.ops.aten.special_ndtri(ref_inp)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_ndtri"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_ndtri(inp)

    _assert_ndtri_dispatch(caplog)
    assert not inp.is_contiguous()
    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_ndtri
@pytest.mark.parametrize("dtype", _NDTRI_UNSUPPORTED_DTYPES)
def test_special_ndtri_unsupported_dtype(dtype):
    inp = _probability_input((8,), dtype)
    with pytest.raises(NotImplementedError):
        with flag_gems.use_gems():
            torch.ops.aten.special_ndtri(inp)
