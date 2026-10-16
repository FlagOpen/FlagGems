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


@pytest.mark.le_scalar_
@pytest.mark.parametrize("shape", [*utils.POINTWISE_SHAPES, (2, 0, 3)])
@pytest.mark.parametrize("scalar", [-1.5, 0.0, 2.25])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_le_scalar_(shape, scalar, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), False)
    reference = ref_inp.le_(scalar)

    result = flag_gems.le_scalar_(inp, scalar)

    assert result is inp
    assert result.data_ptr() == original_pointer
    assert result.dtype == dtype
    utils.gems_assert_equal(result, reference)


@pytest.mark.le_scalar_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_le_scalar_noncontiguous_(dtype):
    torch.manual_seed(0)
    inp = torch.randn((4, 7, 9), dtype=dtype, device=flag_gems.device).transpose(0, 2)
    original_pointer = inp.data_ptr()
    original_stride = inp.stride()
    reference = utils.to_reference(inp.clone(), False).le_(0.5)

    result = flag_gems.le_scalar_(inp, 0.5)

    assert result is inp
    assert result.data_ptr() == original_pointer
    assert result.stride() == original_stride
    utils.gems_assert_equal(result, reference)
