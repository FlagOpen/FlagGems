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


_ATEN_OP = torch.ops.aten._foreach_pow.ScalarAndTensor


def _candidate(*args):
    return flag_gems.foreach_pow_scalar_and_tensor(*args)


def _make_inputs(dtype):
    torch.manual_seed(0)
    shapes = [(16, 8), (7,), (2, 3, 4)]
    make = lambda: [
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
        for shape in shapes
    ]
    exponent_tensors = make()
    return exponent_tensors


@pytest.mark.foreach_pow_scalar_and_tensor
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_foreach_pow_scalar_and_tensor(dtype):
    exponent_tensors = _make_inputs(dtype)
    reference_exponents = [utils.to_reference(t, True) for t in exponent_tensors]
    expected = torch.ops.aten._foreach_pow.ScalarAndTensor(2.0, reference_exponents)
    result = _candidate(2.0, exponent_tensors)

    assert len(result) == len(expected)
    for actual, wanted, source in zip(result, expected, exponent_tensors):
        assert actual.shape == wanted.shape
        assert actual.dtype == wanted.dtype
        assert actual.data_ptr() != source.data_ptr()
        utils.gems_assert_close(actual, wanted, wanted.dtype)


@pytest.mark.foreach_pow_scalar_and_tensor
def test_foreach_pow_scalar_and_tensor_empty_list():
    result = _candidate(2.0, [])
    assert result == []
