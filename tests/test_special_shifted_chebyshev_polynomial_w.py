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


_SUPPORTED_VENDOR = pytest.mark.skipif(
    flag_gems.vendor_name not in ("nvidia", "thead"),
    reason="Only nvidia and thead backends support this op",
)


@_SUPPORTED_VENDOR
@pytest.mark.special_shifted_chebyshev_polynomial_w
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_w_tensor_degree(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    degree = torch.randint(0, 11, shape, dtype=torch.int32, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)

    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_w.default(
        ref_x, ref_degree
    )
    res_out = flag_gems.special_shifted_chebyshev_polynomial_w(x, degree)

    assert res_out.shape == x.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@_SUPPORTED_VENDOR
@pytest.mark.special_shifted_chebyshev_polynomial_w
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_shifted_chebyshev_polynomial_w_degree_boundaries(dtype):
    x = torch.linspace(-1.0, 1.0, 11, dtype=dtype, device=flag_gems.device)
    degree = torch.arange(0, 11, dtype=torch.int32, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)

    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_w.default(
        ref_x, ref_degree
    )
    res_out = flag_gems.special_shifted_chebyshev_polynomial_w(x, degree)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@_SUPPORTED_VENDOR
@pytest.mark.special_shifted_chebyshev_polynomial_w
def test_special_shifted_chebyshev_polynomial_w_rejects_large_degree():
    x = torch.randn(3, dtype=torch.float32, device=flag_gems.device)
    degree = torch.tensor([0, 5, 11], dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(ValueError, match="n must be <= 10"):
        flag_gems.special_shifted_chebyshev_polynomial_w(x, degree)
