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
def _seed_random():
    torch.manual_seed(0)


def _assert_dispatch(caplog):
    expected = (
        f"{utils.gems_log_prefix(flag_gems.special_chebyshev_polynomial_t)} "
        "SPECIAL_CHEBYSHEV_POLYNOMIAL_T"
    )
    assert expected in caplog.text


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_scalar_degree(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 2 - 1
    ref_x = utils.to_reference(x, True)
    degree = 3

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, degree)
    res_out = flag_gems.special_chebyshev_polynomial_t(x, degree)

    assert res_out.shape == x.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_out_of_domain(dtype):
    x = torch.tensor(
        [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_x = utils.to_reference(x, True)
    degree = 3

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, degree)
    res_out = flag_gems.special_chebyshev_polynomial_t(x, degree)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.parametrize("degree", [0, 1, 2, 5])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_scalar_degrees(degree, dtype):
    torch.manual_seed(0)
    x = torch.rand((64,), dtype=dtype, device=flag_gems.device) * 2 - 1
    ref_x = utils.to_reference(x, True)

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, degree)
    res_out = flag_gems.special_chebyshev_polynomial_t(x, degree)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_tensor_degree(dtype):
    x = torch.tensor(
        [-0.75, -0.25, 0.25, 0.75], dtype=dtype, device=flag_gems.device
    )
    degree = torch.tensor([0, 1, 2, 5], dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)

    ref_out = torch.ops.aten.special_chebyshev_polynomial_t.default(
        ref_x, ref_degree
    )
    res_out = flag_gems.special_chebyshev_polynomial_t(x, degree)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_tensor_degree_dispatch(dtype, caplog):
    x = torch.tensor(
        [-0.75, -0.25, 0.25, 0.75], dtype=dtype, device=flag_gems.device
    )
    degree = torch.tensor([0, 1, 2, 5], dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)
    ref_out = torch.ops.aten.special_chebyshev_polynomial_t.default(
        ref_x, ref_degree
    )

    with caplog.at_level(
        "DEBUG", logger="flag_gems.ops.special_chebyshev_polynomial_t"
    ):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_chebyshev_polynomial_t.default(
                x, degree
            )

    _assert_dispatch(caplog)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t
def test_special_chebyshev_polynomial_t_rejects_unsupported_dtype():
    x = torch.randint(0, 4, (16,), dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(ValueError, match="only supports float32/float64"):
        flag_gems.special_chebyshev_polynomial_t(x, 3)
