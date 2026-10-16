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
from . import conftest as cfg

if cfg.QUICK_MODE:
    LIFT_FRESH_SHAPES = [(2, 3)]
else:
    LIFT_FRESH_SHAPES = [(2, 3), (128, 256), (512, 512)]


def _candidate(inp):
    with flag_gems.use_gems():
        return torch.ops.aten.lift_fresh(inp)


@pytest.mark.lift_fresh
@pytest.mark.parametrize("shape", LIFT_FRESH_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lift_fresh_values_and_alias(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    reference_input = utils.to_reference(inp, True)
    reference = torch.ops.aten.lift_fresh(reference_input)
    result = _candidate(inp)

    utils.gems_assert_close(result, reference, dtype)
    assert result is not None
    assert result.shape == inp.shape
    assert result.dtype == inp.dtype
    assert result.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert result.storage_offset() == inp.storage_offset()
    assert result.stride() == inp.stride()


@pytest.mark.lift_fresh
def test_lift_fresh_noncontiguous_write_through():
    torch.manual_seed(0)
    base = torch.randn(6, 8, dtype=torch.float32, device=flag_gems.device)
    inp = base[:, 1::2]
    reference_input = utils.to_reference(inp, True)
    reference = torch.ops.aten.lift_fresh(reference_input)
    result = _candidate(inp)

    utils.gems_assert_close(result, reference, torch.float32)
    assert result is not None
    assert result.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert result.storage_offset() == inp.storage_offset()
    assert result.stride() == inp.stride()

    result[1, 1] = -9.5
    assert inp[1, 1].item() == pytest.approx(-9.5)
    assert base[1, 3].item() == pytest.approx(-9.5)
