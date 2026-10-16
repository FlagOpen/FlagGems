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


_VALID_SHAPES = [*utils.POINTWISE_SHAPES, (257,), (2, 0, 3)]


def _make_valid_input(shape, dtype):
    return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(1.0, 8.0)


def _assert_inplace_result(result, mutated_input, original_pointer, reference, dtype):
    assert result is mutated_input
    assert result.data_ptr() == original_pointer
    assert result.shape == reference.shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.arccosh_
@pytest.mark.parametrize("shape", _VALID_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arccosh_valid_domain_(shape, dtype):
    torch.manual_seed(0)
    inp = _make_valid_input(shape, dtype)
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = ref_inp.arccosh_()

    with flag_gems.use_gems():
        result = inp.arccosh_()

    _assert_inplace_result(result, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.arccosh_
def test_arccosh_boundary_and_special_values_():
    inp = torch.tensor(
        [1.0, 1.0 + 1e-6, 2.0, float("inf"), float("nan")],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = ref_inp.arccosh_()

    with flag_gems.use_gems():
        result = inp.arccosh_()

    _assert_inplace_result(
        result, inp, original_pointer, ref_out, torch.float32
    )


@pytest.mark.inplace
@pytest.mark.arccosh_
def test_arccosh_invalid_domain_produces_nan_():
    inp = torch.tensor(
        [-2.0, -0.0, 0.5, 0.999], dtype=torch.float32, device=flag_gems.device
    )
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = ref_inp.arccosh_()

    with flag_gems.use_gems():
        result = inp.arccosh_()

    _assert_inplace_result(
        result, inp, original_pointer, ref_out, torch.float32
    )


@pytest.mark.inplace
@pytest.mark.arccosh_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arccosh_noncontiguous_(dtype):
    torch.manual_seed(0)
    inp = _make_valid_input((4, 7, 9), dtype).transpose(0, 2)
    original_pointer = inp.data_ptr()
    original_stride = inp.stride()
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = ref_inp.arccosh_()

    with flag_gems.use_gems():
        result = inp.arccosh_()

    assert result.stride() == original_stride
    _assert_inplace_result(result, inp, original_pointer, ref_out, dtype)
