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


_ATEN_OP = torch.ops.aten._foreach_pow.List


def _candidate(*args):
    return flag_gems.foreach_pow_list(*args)


def _make_inputs(dtype):
    torch.manual_seed(0)
    shapes = [(16, 8), (7,), (2, 3, 4)]
    make = lambda: [
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
        for shape in shapes
    ]
    self_tensors = make()
    exponent = make()
    return self_tensors, exponent


@pytest.mark.foreach_pow_list
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_foreach_pow_list(dtype):
    self_tensors, exponent = _make_inputs(dtype)
    reference_self = [utils.to_reference(t.clone(), True) for t in self_tensors]
    reference_exponent = (
        [utils.to_reference(t, True) for t in exponent]
        if exponent and isinstance(exponent, list) and isinstance(exponent[0], torch.Tensor)
        else exponent
    )
    pointers = [t.data_ptr() for t in self_tensors]
    expected = torch.ops.aten._foreach_pow.List(reference_self, reference_exponent)
    result = _candidate(self_tensors, exponent)

    assert len(result) == len(expected)
    for actual, wanted, source in zip(result, expected, self_tensors):
        assert actual.shape == wanted.shape
        assert actual.dtype == wanted.dtype
        assert actual.data_ptr() != source.data_ptr()
        utils.gems_assert_close(actual, wanted, wanted.dtype)


@pytest.mark.foreach_pow_list
def test_foreach_pow_list_empty_list():
    result = _candidate([], [])
    assert result == []


@pytest.mark.foreach_pow_list
def test_foreach_pow_list_rejects_bool_tensor_operand():
    tensors = [torch.tensor([True, False], dtype=torch.bool, device=flag_gems.device)]
    exponents = [torch.tensor([True, True], dtype=torch.bool, device=flag_gems.device)]

    with pytest.raises(NotImplementedError):
        _candidate(tensors, exponents)
    with pytest.raises(NotImplementedError):
        torch.ops.aten._foreach_pow.List(
            [utils.to_reference(t) for t in tensors],
            [utils.to_reference(t) for t in exponents],
        )
