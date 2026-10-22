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

"""Correctness tests for ``aten::empty_strided``.

``empty_strided`` only allocates storage for a requested size/stride pair: it
has no tensor operand, broadcasts nothing and computes no element value. The
default cases compare allocation metadata (shape, stride, dtype, storage size,
offset, contiguity, view/leaf state) against ``torch.ops.aten.empty_strided``,
and element contents only when the deterministic uninitialized-memory fill is
explicitly enabled.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Element types the allocator accepts on this backend (native probe: all of them
# allocate). The capability-gated families reuse the static backend flags from
# accuracy_utils instead of an invented capability list.
_CAPABILITY_GATED_DTYPES = (
    (torch.float64, utils.fp64_is_supported),
    (torch.bfloat16, utils.bf16_is_supported),
    (torch.int64, utils.int64_is_supported),
    (torch.float8_e4m3fn, utils.fp8_is_supported),
    (torch.float8_e5m2, utils.fp8_is_supported),
)
_ALLOCATOR_DTYPES = [
    torch.float32,
    torch.float16,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.bool,
    torch.complex32,
    torch.complex64,
    torch.complex128,
] + [dtype for dtype, supported in _CAPABILITY_GATED_DTYPES if supported]

_LAYOUTS = ("contiguous", "reversed", "padded", "overlap", "zero_stride")


def _contiguous_strides(size):
    strides = [0] * len(size)
    running = 1
    for axis in range(len(size) - 1, -1, -1):
        strides[axis] = running
        running *= size[axis]
    return tuple(strides)


def _strides_for(size, layout):
    contiguous = _contiguous_strides(size)
    if layout == "reversed":
        return tuple(reversed(contiguous))
    if layout == "padded":
        # One padding element per axis: the storage is larger than numel.
        return tuple(step * 2 for step in contiguous)
    if layout == "zero_stride":
        return tuple(0 for _ in contiguous)
    if layout == "overlap":
        # Every axis advances by one element: storage smaller than numel and
        # not dense.
        return tuple(1 for _ in contiguous)
    return contiguous


def _layout_rows(sizes):
    """(size, layout) rows; identical stride tuples are merged per size."""
    rows = []
    for size in sizes:
        seen = set()
        for layout in _LAYOUTS:
            strides = _strides_for(size, layout)
            if strides in seen:
                continue
            seen.add(strides)
            rows.append((size, layout))
    return rows


_GRID_ROWS = _layout_rows(tu.selected_shapes())
# Zero-element allocations are semantic boundaries, so they stay in --quick.
_ZERO_SIZES = ((0,), (0, 3), (3, 0, 2))


def _require_target_dtype(dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype in (
        torch.float64,
        torch.complex128,
    ):
        pytest.skip(
            "Kunlunxin empty_strided narrows requested float64/complex128 "
            "to float32/complex64 on the target device"
        )


def _require_deterministic_fill(dtype):
    if flag_gems.vendor_name in ("ascend", "kunlunxin"):
        pytest.skip(
            f"{flag_gems.vendor_name} empty_strided does not implement "
            "deterministic uninitialized-memory filling"
        )
    if flag_gems.vendor_name == "hygon" and dtype in (
        torch.complex32,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        pytest.skip(
            "Hygon empty_strided deterministic filling is not implemented "
            "for complex32 or float8"
        )


def _assert_allocation(result, reference, size, stride, device, dtype):
    """Compare the allocated storage with the native allocation.

    ``empty_strided`` yields no defined element values, so the observable
    contract is the metadata of the fresh storage plus the absence of
    view/autograd state on it.
    """
    assert result.shape == reference.shape == torch.Size(size)
    assert result.stride() == reference.stride() == tuple(stride)
    assert result.dtype == reference.dtype == dtype
    assert result.numel() == reference.numel()
    assert result.storage_offset() == reference.storage_offset() == 0
    assert result.device == reference.device
    assert result.device.type == torch.device(device).type
    assert result._is_view() is False
    assert not result.is_conj() and not reference.is_conj()
    assert not result.is_neg() and not reference.is_neg()
    assert result.requires_grad is False
    # Native sizes the storage from the requested strides, so it may be smaller
    # than numel (overlapping layout) or larger (padded layout).
    assert result.untyped_storage().nbytes() == reference.untyped_storage().nbytes()
    assert torch.ops.aten.is_non_overlapping_and_dense(
        result
    ) == torch.ops.aten.is_non_overlapping_and_dense(reference)


def _assert_defined_values_equal(result, reference):
    # These side checks compare two allocations explicitly requested on the
    # same device, even with --ref cpu. Check their original ABI before CPU
    # observation; only initialized or deterministic-fill contents reach here.
    assert result.device == reference.device
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape
    assert result.layout == reference.layout == torch.strided
    assert result.is_conj() == reference.is_conj()
    assert result.is_neg() == reference.is_neg()

    def observe(tensor):
        values = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
            tensor.untyped_storage().cpu(),
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
        )
        if tensor.is_conj():
            values = values.conj()
        if tensor.is_neg():
            values = torch._neg_view(values)
        return values

    result_values, reference_values = observe(result), observe(reference)
    tu.assert_result_equal(result_values, reference_values)
    return result_values, reference_values


@pytest.mark.empty_strided
@pytest.mark.parametrize("dtype", _ALLOCATOR_DTYPES)
@pytest.mark.parametrize("size,layout", _GRID_ROWS)
def test_empty_strided_metadata(size, layout, dtype):
    _require_target_dtype(dtype)
    stride = list(_strides_for(size, layout))

    ref_out = torch.ops.aten.empty_strided(
        list(size), stride, dtype=dtype, device=flag_gems.device
    )
    res_out = flag_gems.empty_strided(
        list(size), stride, dtype=dtype, device=flag_gems.device
    )

    _assert_allocation(res_out, ref_out, size, stride, flag_gems.device, dtype)


@pytest.mark.empty_strided
@pytest.mark.parametrize("dtype", _ALLOCATOR_DTYPES)
@pytest.mark.parametrize("size,layout", _layout_rows(_ZERO_SIZES))
def test_empty_strided_zero_element(size, layout, dtype):
    _require_target_dtype(dtype)
    stride = list(_strides_for(size, layout))

    ref_out = torch.ops.aten.empty_strided(
        list(size), stride, dtype=dtype, device=flag_gems.device
    )
    res_out = flag_gems.empty_strided(
        list(size), stride, dtype=dtype, device=flag_gems.device
    )

    _assert_allocation(res_out, ref_out, size, stride, flag_gems.device, dtype)
    assert res_out.numel() == 0
    assert res_out.untyped_storage().nbytes() == 0


_FRESH_SIZE = (5, 7, 3)
_FRESH_STRIDE = (21, 3, 1)


@pytest.mark.empty_strided
@pytest.mark.parametrize("dtype", _ALLOCATOR_DTYPES)
def test_empty_strided_allocates_fresh_storage(dtype):
    _require_target_dtype(dtype)
    stride = list(_FRESH_STRIDE)

    ref_out = torch.ops.aten.empty_strided(
        list(_FRESH_SIZE), stride, dtype=dtype, device=flag_gems.device
    )
    other = flag_gems.empty_strided(
        list(_FRESH_SIZE), stride, dtype=dtype, device=flag_gems.device
    )
    res_out = flag_gems.empty_strided(
        list(_FRESH_SIZE), stride, dtype=dtype, device=flag_gems.device
    )

    _assert_allocation(
        res_out, ref_out, _FRESH_SIZE, _FRESH_STRIDE, flag_gems.device, dtype
    )
    # Each call hands back its own storage instead of a cached buffer.
    assert res_out is not other
    assert res_out.data_ptr() != other.data_ptr()
    encoded = torch.ones(_FRESH_SIZE, dtype=torch.float32, device="cpu").to(dtype)
    filled = torch.empty(_FRESH_SIZE, dtype=dtype, device=flag_gems.device)
    assert filled.dtype == dtype and filled.device == res_out.device
    # Auxiliary writes must not depend on backend ones/copy kernels for FP8
    # or complex values; the allocator's original storage is written directly.
    filled.untyped_storage().copy_(encoded.untyped_storage())
    res_out.untyped_storage().copy_(filled.untyped_storage())
    _assert_defined_values_equal(res_out, filled)


# The out overload resizes the provided buffer to `size` and returns that same
# tensor (native keeps the buffer's strides and storage when the size already
# matches). Buffers stay small, so every variant is part of --quick.
_OUT_ROWS = (
    ((2, 3), (3, 1), (0,), None),
    ((2, 3), (1, 1), (0,), None),
    ((2, 3), (3, 1), (8, 8), None),
    ((2, 3), (3, 1), (2, 3), (1, 2)),
    ((2, 3), (3, 1), (6, 6), (1, 6)),
    ((0, 3), (3, 1), (4, 4), None),
    ((4, 6), (6, 1), (0,), None),
    ((4, 6), (6, 1), (6, 4), None),
    ((4, 6), (6, 1), (3, 4), None),
)


@pytest.mark.empty_strided
@pytest.mark.parametrize("size,stride,buffer_shape,buffer_stride", _OUT_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_strided_out(size, stride, buffer_shape, buffer_stride, dtype):
    _require_target_dtype(dtype)

    def make_buffer():
        if buffer_stride is None:
            return torch.empty(buffer_shape, dtype=dtype, device=flag_gems.device)
        return torch.empty_strided(
            buffer_shape, buffer_stride, dtype=dtype, device=flag_gems.device
        )

    ref_buffer = make_buffer()
    res_buffer = make_buffer()
    ref_device = ref_buffer.device
    res_device = res_buffer.device
    ref_pointer = ref_buffer.untyped_storage().data_ptr()
    res_pointer = res_buffer.untyped_storage().data_ptr()

    torch.ops.aten.empty_strided.out(list(size), list(stride), out=ref_buffer)
    res_out = flag_gems.empty_strided(list(size), list(stride), out=res_buffer)

    # The out overload mutates the supplied tensor in place and returns it.
    assert res_out is res_buffer
    assert res_out.device == res_buffer.device == res_device
    assert ref_buffer.device == ref_device
    assert res_buffer.is_conj() == ref_buffer.is_conj()
    assert res_buffer.is_neg() == ref_buffer.is_neg()
    assert res_buffer.shape == ref_buffer.shape == torch.Size(size)
    assert res_buffer.stride() == ref_buffer.stride()
    assert res_buffer.dtype == ref_buffer.dtype == dtype
    # Resizing reuses the buffer's storage when it is large enough and
    # reallocates otherwise, exactly as the native call does. The requested
    # strides are not honoured on the resize path, so stride equality is
    # checked against the native result above rather than against `stride`.
    assert (res_buffer.untyped_storage().data_ptr() == res_pointer) == (
        ref_buffer.untyped_storage().data_ptr() == ref_pointer
    )
    assert res_buffer.untyped_storage().nbytes() == (
        ref_buffer.untyped_storage().nbytes()
    )


# Each row omits one optional keyword while the other keywords keep the call
# valid; the expected device is the schema default of the omitted argument.
# These are call-form boundaries rather than numerical grids, so all four stay
# in --quick.
_OPTIONAL_ARGUMENTS = (
    ("dtype", flag_gems.device),
    ("layout", flag_gems.device),
    ("device", "cpu"),
    ("pin_memory", flag_gems.device),
)


@pytest.mark.empty_strided
@pytest.mark.parametrize("omitted,expected_device", _OPTIONAL_ARGUMENTS)
def test_empty_strided_optional_argument_default(omitted, expected_device):
    size, stride = (4, 6), [6, 1]
    kwargs = {
        "dtype": torch.float32,
        "layout": torch.strided,
        "device": flag_gems.device,
        "pin_memory": False,
    }
    del kwargs[omitted]

    ref_out = torch.ops.aten.empty_strided(list(size), list(stride), **kwargs)
    res_out = flag_gems.empty_strided(list(size), list(stride), **kwargs)

    _assert_allocation(
        res_out,
        ref_out,
        size,
        stride,
        expected_device,
        kwargs.get("dtype", torch.get_default_dtype()),
    )
    assert res_out.layout == ref_out.layout == torch.strided


_PIN_FLAGS = (True, False)
_PIN_DTYPES = (torch.float32, torch.int8, torch.int64)


@pytest.mark.empty_strided
@pytest.mark.parametrize("pin_memory", _PIN_FLAGS)
@pytest.mark.parametrize("dtype", _PIN_DTYPES)
def test_empty_strided_pin_memory(pin_memory, dtype):
    size, stride = (4, 6), [6, 1]
    # Pinned storage needs a page-locked host allocator, so the schema default
    # CPU device is the one that can honour the flag.
    ref_out = torch.ops.aten.empty_strided(
        list(size), list(stride), dtype=dtype, device="cpu", pin_memory=pin_memory
    )
    res_out = flag_gems.empty_strided(
        list(size), list(stride), dtype=dtype, device="cpu", pin_memory=pin_memory
    )

    _assert_allocation(res_out, ref_out, size, stride, "cpu", dtype)
    assert res_out.is_pinned() == ref_out.is_pinned() == pin_memory


# Pinning exists only for dense host storage: native rejects the accelerator
# device (probe: RuntimeError "Only dense CPU tensors can be pinned"). A CPU
# target accepts the same call, so it is not a negative there.
_PIN_NEGATIVE_DEVICES = [] if flag_gems.device == "cpu" else [flag_gems.device]


@pytest.mark.empty_strided
@pytest.mark.parametrize("device", _PIN_NEGATIVE_DEVICES)
def test_empty_strided_pin_memory_rejected_on_accelerator(device):
    if flag_gems.vendor_name == "kunlunxin":
        pytest.skip(
            "Kunlunxin native empty_strided accepts target pin_memory=True "
            "without pinning rather than rejecting this invalid request"
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.empty_strided(
            [2, 3], [3, 1], dtype=torch.float32, device=device, pin_memory=True
        )


_INVALID_CALLS = (
    pytest.param([2, 3], [3], {}, id="rank-mismatch"),
    pytest.param([-2, 3], [3, 1], {}, id="negative-size"),
    pytest.param([2, 3], [-3, 1], {}, id="negative-stride"),
    pytest.param([2.0, 3], [3, 1], {}, id="non-integer-size"),
    pytest.param(3, [1], {}, id="non-sequence-size"),
    pytest.param([2, 3], 1, {}, id="non-sequence-stride"),
    pytest.param([2, 3], [3, 1], {"layout": torch.sparse_coo}, id="unsupported-layout"),
)


@pytest.mark.empty_strided
@pytest.mark.parametrize("size,stride,extra", _INVALID_CALLS)
def test_empty_strided_invalid_arguments(size, stride, extra):
    if (
        flag_gems.vendor_name == "ascend"
        and isinstance(stride, (tuple, list))
        and any(step < 0 for step in stride)
    ):
        pytest.skip(
            "Ascend native empty_strided does not reject negative strides "
            "and returns a zero-byte allocation instead"
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.empty_strided(
            size, stride, dtype=torch.float32, device=flag_gems.device, **extra
        )


_FILL_FLAGS = (True, False)
_FILL_SIZE = (5, 7, 3)
_FILL_STRIDE = (21, 3, 1)
# The fill matrix asserts defined contents (NaN for float/complex, dtype max for
# integer). Both fill branches are small allocation-contract boundaries.
_FILL_CASES = [(dtype, fill) for fill in _FILL_FLAGS for dtype in _ALLOCATOR_DTYPES]


@pytest.mark.empty_strided
@pytest.mark.parametrize("dtype,fill_uninitialized_memory", _FILL_CASES)
def test_empty_strided_deterministic_fill(dtype, fill_uninitialized_memory):
    _require_target_dtype(dtype)
    if fill_uninitialized_memory:
        _require_deterministic_fill(dtype)
    algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = fill_uninitialized_memory
    try:
        ref_out = torch.ops.aten.empty_strided(
            list(_FILL_SIZE), list(_FILL_STRIDE), dtype=dtype, device=flag_gems.device
        )
        res_out = flag_gems.empty_strided(
            list(_FILL_SIZE), list(_FILL_STRIDE), dtype=dtype, device=flag_gems.device
        )
    finally:
        torch.use_deterministic_algorithms(algorithms_enabled, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill

    _assert_allocation(
        res_out, ref_out, _FILL_SIZE, _FILL_STRIDE, flag_gems.device, dtype
    )
    if fill_uninitialized_memory:
        # With the fill enabled the contents are defined, so the candidate must
        # reproduce them exactly (NaN values compare with equal_nan).
        observed = _assert_defined_values_equal(res_out, ref_out)
        for values in observed:
            if dtype.is_complex:
                assert bool(torch.isnan(values.to(torch.complex128)).all())
            elif dtype.is_floating_point:
                assert bool(torch.isnan(values.float()).all())
            elif dtype == torch.bool:
                assert bool(values.all())
            else:
                assert bool((values == torch.iinfo(dtype).max).all())
