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


_ATEN_OP = torch.ops.aten._foreach_addcdiv.ScalarList


def _candidate(self_tensors, tensor1, tensor2, value):
    return flag_gems.foreach_addcdiv_scalar_list(self_tensors, tensor1, tensor2, value)


def _make_inputs(dtype):
    torch.manual_seed(0)
    shapes = [(16, 8), (7,), (2, 3, 4)]
    make = lambda: [
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
        for shape in shapes
    ]
    self_tensors = make()
    tensor1 = make()
    tensor2 = make()
    value = [0.5] * len(self_tensors)
    return self_tensors, tensor1, tensor2, value


def _reference_value(value):
    return utils.to_reference(value, True) if isinstance(value, torch.Tensor) else value


@pytest.mark.foreach_addcdiv_scalar_list
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_foreach_addcdiv_scalar_list(dtype):
    self_tensors, tensor1, tensor2, value = _make_inputs(dtype)
    reference_self = [utils.to_reference(t.clone(), True) for t in self_tensors]
    reference_tensor1 = [utils.to_reference(t, True) for t in tensor1]
    reference_tensor2 = [utils.to_reference(t, True) for t in tensor2]
    reference_value = _reference_value(value)
    pointers = [t.data_ptr() for t in self_tensors]

    expected = torch.ops.aten._foreach_addcdiv.ScalarList(
        reference_self, reference_tensor1, reference_tensor2, reference_value
    )
    result = _candidate(self_tensors, tensor1, tensor2, value)

    assert len(result) == len(expected) == len(self_tensors)
    for actual, wanted, source in zip(result, expected, self_tensors):
        assert actual.shape == wanted.shape
        assert actual.dtype == wanted.dtype
        assert actual.data_ptr() != source.data_ptr()
        utils.gems_assert_close(actual, wanted, dtype)


@pytest.mark.foreach_addcdiv_scalar_list
def test_foreach_addcdiv_scalar_list_empty_list():
    value = []
    result = _candidate([], [], [], value)
    assert result == []
