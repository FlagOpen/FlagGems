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
from torch.nested._internal.nested_tensor import _nt_view_dummy, _store_val_in_tensor

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="the candidate path requires a target-device nested tensor"
)


def _make_offsets(batch_lengths, dtype=torch.int64):
    offsets = [0]
    for length in batch_lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=dtype, device=flag_gems.device)


def _make_inputs(dtype, batch_lengths, inner_dims):
    torch.manual_seed(0)
    offsets = _make_offsets(batch_lengths)
    values = torch.randn(
        (sum(batch_lengths),) + tuple(inner_dims),
        dtype=dtype,
        device=flag_gems.device,
    )
    return values, offsets


def _assert_nested_equal(result, reference, dtype):
    assert result.is_nested
    assert reference.is_nested
    assert result.shape == reference.shape
    assert result._ragged_idx == reference._ragged_idx
    assert result.values().shape == reference.values().shape
    assert result.values().dtype == dtype
    utils.gems_assert_close(result.values(), reference.values(), dtype)
    utils.gems_assert_equal(result.offsets(), reference.offsets())

    result_lengths = result.lengths()
    reference_lengths = reference.lengths()
    if reference_lengths is None:
        assert result_lengths is None
    else:
        utils.gems_assert_equal(result_lengths, reference_lengths)

    assert set(result._metadata_cache) == set(reference._metadata_cache)
    for key in reference._metadata_cache:
        assert result._metadata_cache[key].shape == reference._metadata_cache[key].shape


def _run(values, offsets, lengths=None, ragged_idx=1, min_seqlen=None, max_seqlen=None):
    dummy = _nt_view_dummy()
    reference = torch.ops.aten._nested_view_from_jagged(
        utils.to_reference(values, True),
        utils.to_reference(offsets, False),
        dummy,
        None if lengths is None else utils.to_reference(lengths, False),
        ragged_idx,
        min_seqlen,
        max_seqlen,
    )
    result = flag_gems._nested_view_from_jagged(
        values, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen
    )
    return result, reference


@pytest.mark.nested_view_from_jagged
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_view_from_jagged_basic(dtype):
    values, offsets = _make_inputs(dtype, [2, 1, 3, 2, 4], (5,))
    result, reference = _run(values, offsets)
    _assert_nested_equal(result, reference, dtype)


@pytest.mark.nested_view_from_jagged
@pytest.mark.parametrize("batch_lengths", [[1], [1, 1, 1], [2, 3, 1, 5, 2, 6]])
def test_nested_view_from_jagged_batch_sizes(batch_lengths):
    values, offsets = _make_inputs(torch.float32, batch_lengths, (4,))
    result, reference = _run(values, offsets)
    _assert_nested_equal(result, reference, torch.float32)


@pytest.mark.nested_view_from_jagged
def test_nested_view_from_jagged_with_lengths():
    batch_lengths = [3, 2, 1, 4, 2]
    values, offsets = _make_inputs(torch.float32, batch_lengths, (5,))
    lengths = torch.tensor(
        batch_lengths, dtype=torch.int64, device=flag_gems.device
    )
    result, reference = _run(values, offsets, lengths=lengths)
    _assert_nested_equal(result, reference, torch.float32)


@pytest.mark.nested_view_from_jagged
def test_nested_view_from_jagged_minmax_seqlen():
    values, offsets = _make_inputs(torch.float32, [3, 2, 1, 4, 2], (5,))
    min_seqlen = _store_val_in_tensor(1)
    max_seqlen = _store_val_in_tensor(4)
    result, reference = _run(
        values,
        offsets,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
    )
    _assert_nested_equal(result, reference, torch.float32)


@pytest.mark.nested_view_from_jagged
def test_nested_view_from_jagged_ragged_idx():
    torch.manual_seed(0)
    batch_lengths = [2, 1, 3]
    offsets = _make_offsets(batch_lengths)
    values = torch.randn(
        4, sum(batch_lengths), 6, dtype=torch.float32, device=flag_gems.device
    )
    result, reference = _run(values, offsets, ragged_idx=2)
    _assert_nested_equal(result, reference, torch.float32)


@pytest.mark.nested_view_from_jagged
def test_nested_view_from_jagged_int32_offsets():
    values, offsets = _make_inputs(torch.float32, [2, 1, 3, 2], (5,))
    offsets = offsets.to(torch.int32)
    result, reference = _run(values, offsets)
    _assert_nested_equal(result, reference, torch.float32)


@pytest.mark.nested_view_from_jagged
def test_nested_view_from_jagged_alias_and_write_through():
    values, offsets = _make_inputs(torch.float32, [2, 1, 3], (4,))
    result, reference = _run(values, offsets)
    _assert_nested_equal(result, reference, torch.float32)

    result_values = result.values()
    assert result is not values
    assert result_values.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
    assert result_values.storage_offset() == values.storage_offset()
    assert result_values.stride() == values.stride()

    result_values[0, 0] = -7.25
    assert values[0, 0].item() == pytest.approx(-7.25)
