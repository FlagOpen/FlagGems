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

_ATEN_OP = torch.ops.aten._foreach_lerp_.List


def _gems_op(*args, **kwargs):
    return flag_gems.foreach_lerp_list_(*args, **kwargs)
SHAPES = [(16, 8), (7,), (2, 3, 4)]
DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _sample(shape, dtype):
    return torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0


def _ref(tensor):
    return utils.to_reference(tensor.clone(), True)


def _assert_list(actual, expected, dtype):
    assert len(actual) == len(expected)
    for result, want in zip(actual, expected):
        assert result.shape == want.shape
        assert result.dtype == want.dtype
        utils.gems_assert_close(result, want, dtype)


@pytest.mark.foreach_lerp_list_
@pytest.mark.parametrize("dtype", DTYPES)
def test_foreach_lerp_list_(dtype):
    torch.manual_seed(0)
    self_tensors = [_sample(shape, dtype) for shape in SHAPES]
    tensors1 = [_sample(shape, dtype) for shape in SHAPES]
    ref_self = [_ref(tensor) for tensor in self_tensors]
    ref_tensors1 = [_ref(tensor) for tensor in tensors1]
    weight = [_sample(shape, dtype) for shape in SHAPES]
    ref_weight = [_ref(tensor) for tensor in weight]
    assert _ATEN_OP(ref_self, ref_tensors1, ref_weight) is None
    pointers = [tensor.data_ptr() for tensor in self_tensors]
    result = _gems_op(self_tensors, tensors1, weight)

    assert result is None
    assert [tensor.data_ptr() for tensor in self_tensors] == pointers
    _assert_list(self_tensors, ref_self, dtype)


@pytest.mark.foreach_lerp_list_
def test_foreach_lerp_list__rejects_mismatched_list_lengths():
    self_tensors = [_sample((8,), torch.float32), _sample((4,), torch.float32)]
    tensors1 = [_sample((8,), torch.float32)]
    with pytest.raises(RuntimeError):
        _gems_op(self_tensors, tensors1, [_sample((8,), torch.float32)])
