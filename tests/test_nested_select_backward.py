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

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="the operator is specific to jagged nested tensors"
)

NESTED_CONFIGS = [
    ([5, 7, 4], 3),
    ([16, 8, 12, 20], 8),
    ([64, 128, 256], 16),
    ([3, 5, 2, 7, 4], 6),
]


def _make_nested(components):
    return torch.nested.nested_tensor(components, layout=torch.jagged)


def _assert_nested_close(result, reference, dtype):
    assert result.is_nested
    assert reference.is_nested
    assert result.values().shape == reference.values().shape
    assert result.values().dtype == dtype
    utils.gems_assert_equal(result.offsets(), reference.offsets())
    utils.gems_assert_close(result.values(), reference.values(), dtype)


@pytest.mark.nested_select_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("sizes,dense_dim", NESTED_CONFIGS)
@pytest.mark.parametrize("index", [0, 1, -1])
def test_nested_select_backward(dtype, sizes, dense_dim, index):
    torch.manual_seed(0)
    components = [
        torch.randn(size, dense_dim, dtype=dtype, device=flag_gems.device)
        for size in sizes
    ]
    nested = _make_nested(components)
    normalized_index = index if index >= 0 else len(sizes) + index
    grad = torch.randn(
        sizes[normalized_index], dense_dim, dtype=dtype, device=flag_gems.device
    )

    reference_nested = _make_nested(
        [utils.to_reference(component, True) for component in components]
    )
    reference = torch.ops.aten._nested_select_backward(
        utils.to_reference(grad, True), reference_nested, 0, index
    )
    result = flag_gems._nested_select_backward(grad, nested, 0, index)

    _assert_nested_close(result, reference, dtype)


@pytest.mark.nested_select_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_select_backward_1d_components(dtype):
    torch.manual_seed(0)
    sizes = [5, 7, 4, 3]
    components = [
        torch.randn(size, dtype=dtype, device=flag_gems.device) for size in sizes
    ]
    nested = _make_nested(components)
    grad = torch.randn(sizes[1], dtype=dtype, device=flag_gems.device)

    reference_nested = _make_nested(
        [utils.to_reference(component, True) for component in components]
    )
    reference = torch.ops.aten._nested_select_backward(
        utils.to_reference(grad, True), reference_nested, 0, 1
    )
    result = flag_gems._nested_select_backward(grad, nested, 0, 1)

    _assert_nested_close(result, reference, dtype)
