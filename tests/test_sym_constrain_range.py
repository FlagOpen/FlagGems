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
from flag_gems.ops.sym_constrain_range import sym_constrain_range as gems_impl

from . import accuracy_utils as utils


_ACCEPTED_CASES = [
    (10, 0, 100),
    (50, 0, None),
    (50, None, 100),
    (42, None, None),
    (-5, -10, 0),
    (0, 0, 0),
    (2**31, 0, 2**31),
]

_REJECTED_CASES = [
    (5, 10, 100),
    (200, 0, 100),
    (-1, 0, None),
]


def _assert_void_matches(res_out, ref_out):
    assert res_out is None
    assert ref_out is None
    # Keep the suite's standard exact-result assertion even though this schema
    # returns void rather than a Tensor.
    utils.gems_assert_equal(
        torch.tensor(res_out is None), torch.tensor(ref_out is None)
    )


@pytest.mark.sym_constrain_range
@pytest.mark.parametrize("size, min_value, max_value", _ACCEPTED_CASES)
def test_sym_constrain_range(size, min_value, max_value):
    kwargs = {
        name: value
        for name, value in (("min", min_value), ("max", max_value))
        if value is not None
    }

    ref_out = torch.ops.aten.sym_constrain_range(size, **kwargs)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.sym_constrain_range(size, **kwargs)

    _assert_void_matches(res_out, ref_out)


@pytest.mark.sym_constrain_range
@pytest.mark.parametrize("size, min_value, max_value", _ACCEPTED_CASES)
def test_sym_constrain_range_direct_implementation(
    size, min_value, max_value, caplog
):
    kwargs = {
        name: value
        for name, value in (("min", min_value), ("max", max_value))
        if value is not None
    }
    ref_out = torch.ops.aten.sym_constrain_range(size, **kwargs)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.sym_constrain_range"):
        res_out = gems_impl(size, **kwargs)

    assert (
        f"{utils.gems_log_prefix(flag_gems.sym_constrain_range)} SYM_CONSTRAIN_RANGE"
        in caplog.text
    )
    _assert_void_matches(res_out, ref_out)


@pytest.mark.sym_constrain_range
@pytest.mark.parametrize("size, min_value, max_value", _REJECTED_CASES)
def test_sym_constrain_range_rejects_out_of_bounds(size, min_value, max_value):
    kwargs = {
        name: value
        for name, value in (("min", min_value), ("max", max_value))
        if value is not None
    }

    with pytest.raises(RuntimeError):
        torch.ops.aten.sym_constrain_range(size, **kwargs)
    with pytest.raises(RuntimeError):
        gems_impl(size, **kwargs)


@pytest.mark.sym_constrain_range
def test_sym_constrain_range_bounds_are_keyword_only():
    with pytest.raises(TypeError):
        gems_impl(5, 0, 10)
