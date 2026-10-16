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
from torch.nested._internal.nested_tensor import _nt_view_dummy

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="the native reference is only available for nested tensors"
)


def _make_inputs(batch, padded_length, trailing_dims, ragged_idx, dtype):
    torch.manual_seed(0)
    shape = [batch] + [1] * (ragged_idx - 1) + [padded_length] + list(trailing_dims)
    padded = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    lengths = [1 + ((index * 7 + 3) % padded_length) for index in range(batch)]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return padded, torch.tensor(offsets, dtype=torch.int32, device=flag_gems.device)


def _assert_nested_close(result, reference, dtype):
    assert result.is_nested
    assert reference.is_nested
    assert result.size() == reference.size()
    assert result.values().shape == reference.values().shape
    assert result.values().dtype == dtype
    utils.gems_assert_close(result.values(), reference.values(), dtype)
    utils.gems_assert_equal(result.offsets(), reference.offsets())


def _run(padded, offsets, ragged_idx=1, sum_s=None):
    dummy = _nt_view_dummy()
    reference = torch.ops.aten._nested_from_padded_tensor(
        utils.to_reference(padded, True),
        utils.to_reference(offsets, False),
        dummy,
        ragged_idx,
        None,
        None,
        sum_s,
    )
    result = flag_gems._nested_from_padded_tensor(
        padded, offsets, dummy, ragged_idx, None, None, sum_s
    )
    return result, reference


@pytest.mark.nested_from_padded_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch,padded_length,dense_dim", [(2, 8, 4), (8, 32, 16), (3, 5, 4)])
def test_nested_from_padded_tensor_3d(dtype, batch, padded_length, dense_dim):
    padded, offsets = _make_inputs(batch, padded_length, (dense_dim,), 1, dtype)
    result, reference = _run(padded, offsets)
    _assert_nested_close(result, reference, dtype)


@pytest.mark.nested_from_padded_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch,padded_length", [(2, 8), (8, 32), (3, 5)])
def test_nested_from_padded_tensor_2d(dtype, batch, padded_length):
    padded, offsets = _make_inputs(batch, padded_length, (), 1, dtype)
    result, reference = _run(padded, offsets)
    _assert_nested_close(result, reference, dtype)


@pytest.mark.nested_from_padded_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch,padded_length,dense_dim", [(2, 8, 4), (8, 32, 16)])
def test_nested_from_padded_tensor_sum_s(dtype, batch, padded_length, dense_dim):
    padded, offsets = _make_inputs(batch, padded_length, (dense_dim,), 1, dtype)
    sum_s = int(offsets[-1].item())
    result, reference = _run(padded, offsets, sum_s=sum_s)
    _assert_nested_close(result, reference, dtype)
    assert result.values().numel() == sum_s * dense_dim


@pytest.mark.nested_from_padded_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_tensor_ragged_idx(dtype):
    torch.manual_seed(0)
    batch, heads, padded_length, dense_dim = 2, 3, 4, 2
    padded = torch.randn(
        batch,
        heads,
        padded_length,
        dense_dim,
        dtype=dtype,
        device=flag_gems.device,
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32, device=flag_gems.device)
    result, reference = _run(padded, offsets, ragged_idx=2)
    _assert_nested_close(result, reference, dtype)
    assert result._ragged_idx == reference._ragged_idx == 2


@pytest.mark.nested_from_padded_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_tensor_cache_invalidation(dtype):
    from flag_gems.ops._nested_from_padded_tensor import _SUM_S_CACHE

    padded, offsets = _make_inputs(4, 16, (3,), 1, dtype)
    first, _ = _run(padded, offsets)
    old_sum_s = first.values().shape[0]
    assert offsets in _SUM_S_CACHE

    offsets[-1] -= 1
    assert offsets._version != _SUM_S_CACHE[offsets][0]

    result, reference = _run(padded, offsets)
    _assert_nested_close(result, reference, dtype)
    assert result.values().shape[0] == old_sum_s - 1
