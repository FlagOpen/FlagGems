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

_ATEN_OP = torch.ops.aten._foreach_copy


def _gems_op(*args, **kwargs):
    return flag_gems.foreach_copy(*args, **kwargs)


@pytest.mark.foreach_copy
@pytest.mark.parametrize(
    "src_dtype", [torch.float16, torch.float32, torch.int64, torch.bool]
)
def test_foreach_copy(src_dtype):
    torch.manual_seed(0)
    destinations = [
        torch.ones((16,), dtype=torch.float16, device=flag_gems.device),
        torch.ones((3, 5), dtype=torch.float16, device=flag_gems.device),
    ]
    sources = [
        torch.randint(0, 2, destination.shape, device=flag_gems.device).to(
            src_dtype
        )
        if not src_dtype.is_floating_point
        else torch.rand(
            destination.shape, dtype=src_dtype, device=flag_gems.device
        )
        for destination in destinations
    ]
    ref_destinations = [utils.to_reference(tensor.clone()) for tensor in destinations]
    ref_sources = [utils.to_reference(tensor) for tensor in sources]
    expected = _ATEN_OP(ref_destinations, ref_sources)
    pointers = [tensor.data_ptr() for tensor in destinations]
    actual = _gems_op(destinations, sources)

    assert all(
        output.data_ptr() != destination.data_ptr()
        for output, destination in zip(actual, destinations)
    )
    assert [tensor.data_ptr() for tensor in destinations] == pointers
    assert len(actual) == len(expected)
    for result, want in zip(actual, expected):
        assert result.shape == want.shape
        assert result.dtype == want.dtype
        utils.gems_assert_equal(result, want)


@pytest.mark.foreach_copy
def test_foreach_copy_noncontiguous():
    destinations = [
        torch.ones((6, 8), device=flag_gems.device).transpose(0, 1)
    ]
    sources = [
        torch.arange(48, dtype=torch.float32, device=flag_gems.device).reshape(
            8, 6
        )
    ]
    ref_destinations = [utils.to_reference(tensor.clone()) for tensor in destinations]
    ref_sources = [utils.to_reference(tensor) for tensor in sources]
    expected = _ATEN_OP(ref_destinations, ref_sources)
    result = _gems_op(destinations, sources)

    actual = result
    utils.gems_assert_equal(actual[0], expected[0])
