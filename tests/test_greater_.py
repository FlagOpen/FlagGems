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


_EQUAL_SHAPES = [*utils.POINTWISE_SHAPES, (257,), (2, 0, 3)]
_BROADCAST_CASES = [
    pytest.param((7, 13), (1, 13), id="leading-dimension"),
    pytest.param((2, 3, 5), (1, 3, 1), id="multiple-dimensions"),
    pytest.param((2, 0, 3), (1, 0, 1), id="zero-size"),
]
_SCALARS = (-1.5, 0.0, 2.25)


def _make_input(shape, dtype):
    return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(-4.0, 4.0)


def _assert_inplace_result(result, mutated_input, original_pointer, reference, dtype):
    assert result is mutated_input
    assert result.data_ptr() == original_pointer
    assert result.shape == reference.shape
    assert result.dtype == dtype
    utils.gems_assert_equal(result, reference)


@pytest.mark.inplace
@pytest.mark.greater_
@pytest.mark.parametrize("shape", _EQUAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_tensor_equal_shape_(shape, dtype):
    torch.manual_seed(0)
    inp = _make_input(shape, dtype)
    other = _make_input(shape, dtype)
    other_before = other.clone()
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone())
    ref_other = utils.to_reference(other)
    ref_out = ref_inp.greater_(ref_other)

    with flag_gems.use_gems():
        result = inp.greater_(other)

    _assert_inplace_result(result, inp, original_pointer, ref_out, dtype)
    utils.gems_assert_equal(other, other_before)


@pytest.mark.inplace
@pytest.mark.greater_
@pytest.mark.parametrize("self_shape,other_shape", _BROADCAST_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_tensor_broadcast_(self_shape, other_shape, dtype):
    torch.manual_seed(0)
    inp = _make_input(self_shape, dtype)
    other = _make_input(other_shape, dtype)
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone())
    ref_other = utils.to_reference(other)
    ref_out = ref_inp.greater_(ref_other)

    with flag_gems.use_gems():
        result = inp.greater_(other)

    _assert_inplace_result(result, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.greater_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", _SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_scalar_(shape, scalar, dtype):
    torch.manual_seed(0)
    inp = _make_input(shape, dtype)
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.greater_(scalar)

    with flag_gems.use_gems():
        result = inp.greater_(scalar)

    _assert_inplace_result(result, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.greater_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_greater_noncontiguous_(dtype):
    torch.manual_seed(0)
    inp = _make_input((4, 7, 9), dtype).transpose(0, 2)
    other = _make_input(inp.shape, dtype)
    original_pointer = inp.data_ptr()
    original_stride = inp.stride()
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp.clone())
    ref_other = utils.to_reference(other)
    ref_out = ref_inp.greater_(ref_other)

    with flag_gems.use_gems():
        result = inp.greater_(other)

    assert result.stride() == original_stride
    _assert_inplace_result(result, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.greater_
def test_greater_fp64_precision_():
    inp = torch.tensor(
        [16777216.5], dtype=torch.float64, device=flag_gems.device
    )
    other = torch.tensor(
        [16777216.4], dtype=torch.float64, device=flag_gems.device
    )
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone())
    ref_other = utils.to_reference(other)
    ref_out = ref_inp.greater_(ref_other)

    with flag_gems.use_gems():
        result = inp.greater_(other)

    _assert_inplace_result(
        result, inp, original_pointer, ref_out, torch.float64
    )


@pytest.mark.inplace
@pytest.mark.greater_
def test_greater_rejects_nonbroadcastable_shape_():
    inp = torch.ones((2, 3), device=flag_gems.device)
    other = torch.ones((4,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            inp.greater_(other)
