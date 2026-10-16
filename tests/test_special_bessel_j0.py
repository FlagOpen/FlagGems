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


_SKIP_FP64_ON_KUNLUNXIN = pytest.mark.skipif(
    flag_gems.runtime.device.vendor_name == "kunlunxin",
    reason="Kunlunxin XPU has no real float64 (silently downgraded to float32)",
)
_FLOAT_DTYPES = [
    torch.float32,
    pytest.param(torch.float64, marks=_SKIP_FP64_ON_KUNLUNXIN),
]
_POINTWISE_SHAPES = [(128,), (512, 256), (2, 128, 128), (0,)]
_NO_DISPATCH_LOG_VENDORS = {"iluvatar", "metax"}


def _run_special_bessel_j0(inp, caplog):
    logger_name = flag_gems.special_bessel_j0.__module__
    with caplog.at_level("DEBUG", logger=logger_name):
        with flag_gems.use_gems():
            result = torch.special.bessel_j0(inp)
    if flag_gems.vendor_name in _NO_DISPATCH_LOG_VENDORS:
        assert logger_name.startswith("flag_gems.")
    else:
        assert "SPECIAL_BESSEL_J0" in caplog.text
    return result


@pytest.mark.special_bessel_j0
@pytest.mark.parametrize("shape", _POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_bessel_j0(shape, dtype, caplog):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.bessel_j0(ref_inp)

    res_out = _run_special_bessel_j0(inp, caplog)

    assert res_out.shape == inp.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_bessel_j0
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_bessel_j0_noncontiguous(dtype, caplog):
    torch.manual_seed(0)
    inp = torch.randn((16, 32), dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.bessel_j0(ref_inp)

    res_out = _run_special_bessel_j0(inp, caplog)

    assert not inp.is_contiguous()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_bessel_j0
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_bessel_j0_special_values(dtype, caplog):
    inp = torch.tensor(
        [float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.bessel_j0(ref_inp)

    res_out = _run_special_bessel_j0(inp, caplog)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
