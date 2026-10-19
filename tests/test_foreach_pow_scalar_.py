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


_ATEN_OP = torch.ops.aten._foreach_pow_.Scalar


def _candidate(*args):
    with flag_gems.use_gems():
        return _ATEN_OP(*args)


def _make_inputs(dtype):
    torch.manual_seed(0)
    shapes = [(16, 8), (7,), (2, 3, 4)]
    make = lambda: [
        torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
        for shape in shapes
    ]
    self_tensors = make()
    exponent = 0.5
    return self_tensors, exponent


@pytest.mark.foreach_pow_scalar_
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_foreach_pow_scalar_(dtype):
    self_tensors, exponent = _make_inputs(dtype)
    reference_self = [utils.to_reference(t.clone(), True) for t in self_tensors]
    reference_exponent = (
        [utils.to_reference(t, True) for t in exponent]
        if exponent and isinstance(exponent, list) and isinstance(exponent[0], torch.Tensor)
        else exponent
    )
    pointers = [t.data_ptr() for t in self_tensors]
    expected = torch.ops.aten._foreach_pow_.Scalar(reference_self, reference_exponent)
    result = _candidate(self_tensors, exponent)

    assert result is None
    assert [t.data_ptr() for t in self_tensors] == pointers
    for actual, wanted in zip(self_tensors, reference_self):
        utils.gems_assert_close(actual, wanted, dtype)


@pytest.mark.foreach_pow_scalar_
def test_foreach_pow_scalar__empty_list():
    result = _candidate([], 0.5)
    assert result is None
