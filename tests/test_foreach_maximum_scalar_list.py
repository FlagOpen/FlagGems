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


_ATEN_OP = torch.ops.aten._foreach_maximum.ScalarList


def _gems_op(*args, **kwargs):
    with flag_gems.use_gems():
        return _ATEN_OP(*args, **kwargs)


_SHAPES = ((4, 7), (3, 5))


def _make_tensor(shape, dtype, noncontiguous, *, positive=False):
    if noncontiguous:
        tensor = torch.randn(
            (shape[1], shape[0]), dtype=dtype, device=flag_gems.device
        ).transpose(0, 1)
    else:
        tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if positive:
        tensor = tensor.abs().add_(0.5)
    return tensor


def _make_inputs(dtype, noncontiguous):
    torch.manual_seed(0)
    tensors = [
        _make_tensor(shape, dtype, noncontiguous) for shape in _SHAPES
    ]
    other = [1.5, 2.0]
    return tensors, other, {}


def _reference_other(other):
    return list(other)


@pytest.mark.foreach_maximum_scalar_list
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "noncontiguous", [False, True], ids=["contiguous", "noncontiguous"]
)
def test_foreach_maximum_scalar_list_accuracy(dtype, noncontiguous):
    tensors, other, kwargs = _make_inputs(dtype, noncontiguous)
    ref_tensors = [utils.to_reference(value.clone(), True) for value in tensors]
    ref_other = _reference_other(other)
    pointers = [value.data_ptr() for value in tensors]
    strides = [value.stride() for value in tensors]

    reference_result = _ATEN_OP(ref_tensors, ref_other, **kwargs)
    result = _gems_op(tensors, other, **kwargs)

    assert isinstance(result, (list, tuple))
    assert len(result) == len(reference_result) == len(tensors)
    for out, expected, inp in zip(result, reference_result, tensors):
        assert out.shape == expected.shape
        assert out.dtype == dtype
        assert out.data_ptr() != inp.data_ptr()
        utils.gems_assert_close(out, expected, dtype)


@pytest.mark.foreach_maximum_scalar_list
def test_foreach_maximum_scalar_list_rejects_mismatched_list_length():
    tensors, other, kwargs = _make_inputs(torch.float32, False)
    with pytest.raises(RuntimeError):
        _gems_op(tensors, other[:-1], **kwargs)
