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
from . import test_utils as tu


def _candidate(sizes):
    if not isinstance(sizes, torch.Tensor):
        return flag_gems._nested_compute_contiguous_strides_offsets(sizes)
    storage = sizes.untyped_storage()
    identity = (
        storage._cdata,
        storage.data_ptr(),
        storage.nbytes(),
        sizes.device,
        sizes.dtype,
        sizes.shape,
        sizes.stride(),
        sizes.storage_offset(),
        sizes.is_neg(),
        sizes.is_conj(),
        sizes.requires_grad,
    )
    before = (
        torch.empty(0, dtype=torch.uint8, device="cpu")
        .set_(storage, 0, (storage.nbytes(),), (1,))
        .clone()
    )
    try:
        result = flag_gems._nested_compute_contiguous_strides_offsets(sizes)
    finally:
        after_storage = sizes.untyped_storage()
        assert (
            after_storage._cdata,
            after_storage.data_ptr(),
            after_storage.nbytes(),
            sizes.device,
            sizes.dtype,
            sizes.shape,
            sizes.stride(),
            sizes.storage_offset(),
            sizes.is_neg(),
            sizes.is_conj(),
            sizes.requires_grad,
        ) == identity
        after = torch.empty(0, dtype=torch.uint8, device="cpu").set_(
            after_storage, 0, (after_storage.nbytes(),), (1,)
        )
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert isinstance(result, tuple) and len(result) == 2
    for value in result:
        assert isinstance(value, torch.Tensor)
        assert value.device == torch.device("cpu") and value.dtype == torch.int64
    return result


# Derive contiguous strides and cumulative storage offsets from CPU int64 sizes.
# Keep inputs non-empty and on CPU: the current reference crashes otherwise.
# Bound size products below 2**20 and total offsets below 2**29 to avoid its
# int32 truncation. The current reference has no kernel for the .out overload.
_NUM_TENSORS = [1, 3, 8, 64, 512]

_NUM_DIMS = [1, 2, 3, 5]

_SIZE_PATTERNS = ["uniform", "with_zero", "all_ones", "all_zero", "wide"]

_SIZE_MODULUS = 9

_UNSUPPORTED_DTYPES = [
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

_VALUE_RANGE_LAYOUTS = tu.selected_cases([(8, 2), (64, 3)], quick=[(8, 2)])


def _make_nested_size(num_tensors, num_dims, pattern, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    if pattern == "uniform":
        # Typical ragged batch: every sub-tensor is non-empty with varied dims.
        return torch.randint(
            1,
            9,
            (num_tensors, num_dims),
            dtype=torch.int64,
            device="cpu",
            generator=gen,
        )
    if pattern == "with_zero":
        # Some sub-tensors are empty (zero-size dims); offsets may repeat.
        return torch.randint(
            0,
            9,
            (num_tensors, num_dims),
            dtype=torch.int64,
            device="cpu",
            generator=gen,
        )
    if pattern == "all_ones":
        # Minimal positive sizes: every product is 1.
        return torch.ones((num_tensors, num_dims), dtype=torch.int64, device="cpu")
    if pattern == "all_zero":
        # Degenerate batch: every per-row product is 0, so every offset is 0
        # and every stride row is [0, ..., 0, 1] (innermost stride prod(()) = 1).
        return torch.zeros((num_tensors, num_dims), dtype=torch.int64, device="cpu")
    if pattern == "wide":
        # Wider value range than "uniform"; the bound keeps every per-row
        # product below 2**20 even in the worst case.
        bound = max(2, 2 ** (20 // num_dims))
        return torch.randint(
            1,
            bound,
            (num_tensors, num_dims),
            dtype=torch.int64,
            device="cpu",
            generator=gen,
        )
    raise ValueError(f"Unknown size pattern: {pattern!r}")


def _make_sizes_from_range(num_tensors, num_dims, value_range, seed=0):
    # Map sampled int64 values to valid small extents on CPU, where the reference reads them.
    raw = tu.make_input(torch.int64, (num_tensors, num_dims), value_range)
    return raw.cpu().remainder(_SIZE_MODULUS)


@pytest.mark.underscore_nested_compute_contiguous_strides_offsets
@pytest.mark.parametrize("pattern", _SIZE_PATTERNS)
@pytest.mark.parametrize("num_dims", _NUM_DIMS)
@pytest.mark.parametrize("num_tensors", _NUM_TENSORS)
def test__nested_compute_contiguous_strides_offsets(num_tensors, num_dims, pattern):
    sizes = _make_nested_size(num_tensors, num_dims, pattern)
    ref_sizes = tu.to_reference(sizes)

    (
        ref_strides,
        ref_offsets,
    ) = torch.ops.aten._nested_compute_contiguous_strides_offsets(ref_sizes)
    res_strides, res_offsets = _candidate(sizes)

    utils.gems_assert_equal(res_strides, ref_strides)
    utils.gems_assert_equal(res_offsets, ref_offsets)


@pytest.mark.underscore_nested_compute_contiguous_strides_offsets
@pytest.mark.parametrize("layout", _VALUE_RANGE_LAYOUTS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__nested_compute_contiguous_strides_offsets_value_ranges(layout, value_range):
    num_tensors, num_dims = layout
    sizes = _make_sizes_from_range(num_tensors, num_dims, value_range)
    ref_sizes = tu.to_reference(sizes)

    (
        ref_strides,
        ref_offsets,
    ) = torch.ops.aten._nested_compute_contiguous_strides_offsets(ref_sizes)
    res_strides, res_offsets = _candidate(sizes)

    utils.gems_assert_equal(res_strides, ref_strides)
    utils.gems_assert_equal(res_offsets, ref_offsets)


@pytest.mark.underscore_nested_compute_contiguous_strides_offsets
def test__nested_compute_contiguous_strides_offsets_known_layout():
    sizes = torch.tensor(
        [[2, 3], [4, 3], [1, 3], [3, 3]], dtype=torch.int64, device="cpu"
    )

    (
        ref_strides,
        ref_offsets,
    ) = torch.ops.aten._nested_compute_contiguous_strides_offsets(
        tu.to_reference(sizes)
    )
    res_strides, res_offsets = _candidate(sizes)

    utils.gems_assert_equal(res_strides, ref_strides)
    utils.gems_assert_equal(res_offsets, ref_offsets)


@pytest.mark.underscore_nested_compute_contiguous_strides_offsets
def test__nested_compute_contiguous_strides_offsets_invalid_ndim():
    bad = torch.ones(4, dtype=torch.int64, device="cpu")
    with pytest.raises(IndexError):
        torch.ops.aten._nested_compute_contiguous_strides_offsets(bad)
    with pytest.raises((IndexError, RuntimeError, ValueError)):
        _candidate(bad)


@pytest.mark.underscore_nested_compute_contiguous_strides_offsets
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test__nested_compute_contiguous_strides_offsets_rejects_non_int64(dtype):
    bad = torch.zeros((4, 3), dtype=dtype, device="cpu")
    with pytest.raises(RuntimeError):
        torch.ops.aten._nested_compute_contiguous_strides_offsets(bad)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _candidate(bad)


@pytest.mark.underscore_nested_compute_contiguous_strides_offsets
def test__nested_compute_contiguous_strides_offsets_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._nested_compute_contiguous_strides_offsets(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _candidate(3.14)
