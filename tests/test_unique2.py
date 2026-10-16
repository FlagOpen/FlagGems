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


def _reference_input(inp):
    if inp.dtype in utils.ALL_FLOAT_DTYPES:
        return utils.to_reference(inp, True)
    return utils.to_reference(inp, False)


def _assert_unique2(result, reference, dtype):
    output, inverse_indices, counts = result
    ref_output, ref_inverse_indices, ref_counts = reference
    assert output.dtype == dtype
    assert inverse_indices.dtype == torch.int64
    assert counts.dtype == torch.int64
    if dtype in utils.ALL_FLOAT_DTYPES:
        utils.gems_assert_close(output, ref_output, dtype, equal_nan=True)
    else:
        utils.gems_assert_equal(output, ref_output)
    utils.gems_assert_equal(inverse_indices, ref_inverse_indices)
    utils.gems_assert_equal(counts, ref_counts)


@pytest.mark.unique2
@pytest.mark.parametrize("shape", [*utils.POINTWISE_SHAPES, (8193,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize("sorted_", [False, True])
@pytest.mark.parametrize(
    "return_inverse,return_counts",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_unique2(shape, dtype, sorted_, return_inverse, return_counts):
    torch.manual_seed(0)
    inp = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    before = inp.clone()
    ref_inp = _reference_input(inp)
    reference = torch._unique2(
        ref_inp,
        sorted=sorted_,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    with flag_gems.use_gems():
        result = torch._unique2(
            inp,
            sorted=sorted_,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )

    _assert_unique2(result, reference, dtype)
    utils.gems_assert_equal(inp, before)


@pytest.mark.unique2
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize(
    "return_inverse,return_counts",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_unique2_empty(dtype, return_inverse, return_counts):
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    reference = torch._unique2(
        _reference_input(inp),
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    with flag_gems.use_gems():
        result = torch._unique2(
            inp,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )

    assert all(item.numel() == 0 for item in result)
    _assert_unique2(result, reference, dtype)


@pytest.mark.unique2
@pytest.mark.parametrize("all_unique", [False, True])
def test_unique2_integer_edge_cases(all_unique):
    inp = (
        torch.arange(100, dtype=torch.int64, device=flag_gems.device)
        if all_unique
        else torch.ones(100, dtype=torch.int64, device=flag_gems.device)
    )
    reference = torch._unique2(
        _reference_input(inp), return_inverse=True, return_counts=True
    )

    with flag_gems.use_gems():
        result = torch._unique2(inp, return_inverse=True, return_counts=True)

    _assert_unique2(result, reference, torch.int64)


@pytest.mark.unique2
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.float64,
            marks=pytest.mark.skipif(
                not utils.fp64_is_supported, reason="device does not support fp64"
            ),
        ),
    ],
)
def test_unique2_float_special_values(dtype):
    values = [
        1.5,
        -2.25,
        0.0,
        -0.0,
        float("inf"),
        -2.25,
        float("nan"),
        1.5,
        float("nan"),
        float("-inf"),
        float("nan"),
        3.75,
    ]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(inp)
    ref_output, _, ref_counts = torch._unique2(
        ref_inp, return_inverse=True, return_counts=True
    )

    with flag_gems.use_gems():
        output, inverse_indices, counts = torch._unique2(
            inp, return_inverse=True, return_counts=True
        )

    utils.gems_assert_close(output, ref_output, dtype, equal_nan=True)
    utils.gems_assert_equal(counts, ref_counts)
    utils.gems_assert_close(
        output[inverse_indices], ref_inp, dtype, equal_nan=True
    )
