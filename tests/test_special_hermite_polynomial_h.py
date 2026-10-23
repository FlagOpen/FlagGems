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

_FLOAT_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    _FLOAT_DTYPES.append(torch.float64)


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


def _skip_unsupported_cambricon_float64(dtype):
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip(reason="Issue #5253: float64 is not supported on Cambricon")


def _assert_gems_call(caplog):
    expected = (
        f"{utils.gems_log_prefix(flag_gems.special_hermite_polynomial_h)} "
        "SPECIAL_HERMITE_POLYNOMIAL_H"
    )
    assert expected in caplog.text


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.hermite_polynomial_h reference only supports float32 and float64
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_hermite_polynomial_h_tensor_degree(shape, dtype):
    _skip_unsupported_cambricon_float64(dtype)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    degree = torch.randint(0, 10, (), dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)

    ref_out = torch.ops.aten.special_hermite_polynomial_h.default(ref_x, ref_degree)
    res_out = flag_gems.special_hermite_polynomial_h(x, degree)

    assert res_out.shape == x.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.hermite_polynomial_h reference only supports float32 and float64
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_special_hermite_polynomial_h_scalar_degree(shape, dtype):
    _skip_unsupported_cambricon_float64(dtype)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    degree = 9
    ref_x = utils.to_reference(x, True)

    ref_out = torch.ops.aten.special_hermite_polynomial_h.n_scalar(ref_x, degree)
    res_out = flag_gems.special_hermite_polynomial_h(x, degree)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("overload", ["default", "n_scalar"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_hermite_polynomial_h_direct_call(overload, dtype, caplog):
    x = torch.tensor([-1.0, -0.25, 0.5, 1.0], dtype=dtype, device=flag_gems.device)
    degree = (
        torch.tensor([0, 1, 5, 9], dtype=torch.int64, device=flag_gems.device)
        if overload == "default"
        else 9
    )
    ref_x = utils.to_reference(x, True)
    ref_degree = (
        utils.to_reference(degree, False)
        if isinstance(degree, torch.Tensor)
        else degree
    )
    reference_op = getattr(torch.ops.aten.special_hermite_polynomial_h, overload)
    ref_out = reference_op(ref_x, ref_degree)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_hermite_polynomial_h"):
        res_out = flag_gems.special_hermite_polynomial_h(x, degree)

    _assert_gems_call(caplog)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
@pytest.mark.parametrize("degree", [-1, 10])
def test_special_hermite_polynomial_h_rejects_scalar_degree(dtype, degree):
    _skip_unsupported_cambricon_float64(dtype)
    x = torch.randn((4, 4), dtype=dtype, device=flag_gems.device)

    with pytest.raises(ValueError, match=r"only supports n in \[0, 9\]"):
        flag_gems.special_hermite_polynomial_h(x, degree)


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("degree", [-1, 10])
def test_special_hermite_polynomial_h_rejects_tensor_degree(degree):
    x = torch.randn((4, 4), dtype=torch.float32, device=flag_gems.device)
    bad_degree = torch.tensor(degree, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(ValueError, match=r"only supports n in \[0, 9\]"):
        flag_gems.special_hermite_polynomial_h(x, bad_degree)
