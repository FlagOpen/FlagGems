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
from . import conftest as cfg


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float64]


def _assert_close(result, reference, dtype):
    atol = 5e-3 if dtype == torch.float32 else 1e-8
    utils.gems_assert_close(result, reference, dtype, atol=atol, equal_nan=True)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_shifted_chebyshev_polynomial_t_tensor_degree(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    degree = torch.randint(0, 10, shape, dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)

    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t.default(
        ref_x, ref_degree
    )
    res_out = flag_gems.special_shifted_chebyshev_polynomial_t(x, degree)

    assert res_out.shape == x.shape
    assert res_out.dtype == dtype
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_shifted_chebyshev_polynomial_t_scalar_degree(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    degree = 3
    ref_x = utils.to_reference(x, True)

    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t.n_scalar(
        ref_x, degree
    )
    res_out = flag_gems.special_shifted_chebyshev_polynomial_t(x, degree)

    _assert_close(res_out, ref_out, dtype)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_shifted_chebyshev_polynomial_t_registered_dispatch(dtype, caplog):
    x = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=dtype, device=flag_gems.device)
    degree = torch.tensor([0, 1, 2, 9], dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)
    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t.default(
        ref_x, ref_degree
    )

    with caplog.at_level(
        "DEBUG", logger="flag_gems.ops.special_shifted_chebyshev_polynomial_t"
    ):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t.default(
                x, degree
            )

    expected = (
        f"{utils.gems_log_prefix(flag_gems.special_shifted_chebyshev_polynomial_t)} "
        "SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_T"
    )
    assert expected in caplog.text
    _assert_close(res_out, ref_out, dtype)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_special_shifted_chebyshev_polynomial_t_rejects_unsupported_dtype(dtype):
    x = torch.rand((16,), dtype=dtype, device=flag_gems.device)

    with pytest.raises(ValueError, match="has no Half/BFloat16 implementation"):
        flag_gems.special_shifted_chebyshev_polynomial_t(x, 3)
