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


SHAPES = [(3, 4), (5, 3), (2, 3, 4), (4,), (8, 16), (2, 8, 16)]
CASES = [(shape, dim) for shape in SHAPES for dim in range(len(shape))]


@pytest.mark.index_select_backward
@pytest.mark.parametrize("shape,dim", CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_select_backward(shape, dim, dtype):
    dim_size_out = shape[dim] + 2
    self_sizes = list(shape)
    self_sizes[dim] = dim_size_out
    self_sizes = tuple(self_sizes)
    torch.manual_seed(0)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, dim_size_out, (shape[dim],), device=flag_gems.device)
    ref_out = torch.ops.aten.index_select_backward(
        utils.to_reference(grad, True), self_sizes, dim, utils.to_reference(index, False)
    )
    result = flag_gems.index_select_backward(grad, self_sizes, dim, index)
    assert result.shape == self_sizes
    assert result.dtype == dtype
    utils.gems_assert_close(result, ref_out, dtype)


@pytest.mark.index_select_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_select_backward_repeated_indices(dtype):
    torch.manual_seed(0)
    grad = torch.randn(4, dtype=dtype, device=flag_gems.device)
    index = torch.tensor([0, 2, 2, 5], device=flag_gems.device)
    ref_out = torch.ops.aten.index_select_backward(
        utils.to_reference(grad, True), (6,), 0, utils.to_reference(index, False)
    )
    result = flag_gems.index_select_backward(grad, (6,), 0, index)
    utils.gems_assert_close(result, ref_out, dtype)
