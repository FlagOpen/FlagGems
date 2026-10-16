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
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)


def _assert_new_result_without_input_mutation(result, *inputs):
    result_storage = result.untyped_storage().data_ptr()
    for tensor, snapshot, pointer in inputs:
        assert tensor.data_ptr() == pointer
        utils.gems_assert_equal(tensor, snapshot)
        assert result_storage != tensor.untyped_storage().data_ptr()


@_SUPPORTED_VENDOR
@pytest.mark.special_chebyshev_polynomial_w
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_w_scalar_degree(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 2 - 1
    x_before = x.clone()
    x_pointer = x.data_ptr()
    ref_x = utils.to_reference(x, True)
    degree = 3

    ref_out = torch.special.chebyshev_polynomial_w(ref_x, degree)
    result = flag_gems.special_chebyshev_polynomial_w(x, degree)

    assert result.shape == x.shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, ref_out, dtype, equal_nan=True)
    _assert_new_result_without_input_mutation(
        result, (x, x_before, x_pointer)
    )


@_SUPPORTED_VENDOR
@pytest.mark.special_chebyshev_polynomial_w
@pytest.mark.parametrize("degree", [0, 1, 2, 5])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_w_scalar_degrees(degree, dtype):
    x = torch.rand((64,), dtype=dtype, device=flag_gems.device) * 2 - 1
    x_before = x.clone()
    x_pointer = x.data_ptr()
    ref_x = utils.to_reference(x, True)

    ref_out = torch.special.chebyshev_polynomial_w(ref_x, degree)
    result = flag_gems.special_chebyshev_polynomial_w(x, degree)

    utils.gems_assert_close(result, ref_out, dtype, equal_nan=True)
    _assert_new_result_without_input_mutation(
        result, (x, x_before, x_pointer)
    )


@_SUPPORTED_VENDOR
@pytest.mark.special_chebyshev_polynomial_w
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_w_tensor_degree(dtype):
    x = torch.tensor(
        [-0.75, -0.25, 0.25, 0.75], dtype=dtype, device=flag_gems.device
    )
    degree = torch.tensor(
        [0, 1, 2, 5], dtype=torch.int64, device=flag_gems.device
    )
    x_before = x.clone()
    degree_before = degree.clone()
    x_pointer = x.data_ptr()
    degree_pointer = degree.data_ptr()
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)

    ref_out = torch.ops.aten.special_chebyshev_polynomial_w.default(ref_x, ref_degree)
    result = flag_gems.special_chebyshev_polynomial_w(x, degree)

    utils.gems_assert_close(result, ref_out, dtype, equal_nan=True)
    _assert_new_result_without_input_mutation(
        result,
        (x, x_before, x_pointer),
        (degree, degree_before, degree_pointer),
    )


@_SUPPORTED_VENDOR
@pytest.mark.special_chebyshev_polynomial_w
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_w_out_of_domain(dtype):
    x = torch.tensor(
        [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    x_before = x.clone()
    x_pointer = x.data_ptr()
    ref_x = utils.to_reference(x, True)

    ref_out = torch.special.chebyshev_polynomial_w(ref_x, 3)
    result = flag_gems.special_chebyshev_polynomial_w(x, 3)

    utils.gems_assert_close(result, ref_out, dtype, equal_nan=True)
    _assert_new_result_without_input_mutation(
        result, (x, x_before, x_pointer)
    )


@pytest.mark.special_chebyshev_polynomial_w
def test_special_chebyshev_polynomial_w_rejects_unsupported_dtype():
    x = torch.randint(0, 4, (16,), dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(ValueError, match="only supports float32/float64"):
        flag_gems.special_chebyshev_polynomial_w(x, 3)
