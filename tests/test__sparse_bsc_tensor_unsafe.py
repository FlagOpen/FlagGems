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

"""Correctness tests for ``aten::_sparse_bsc_tensor_unsafe``.

The operator is a host-side factory: it installs column-block pointers,
block-row indices and values in a BSC TensorImpl and validates none of them, so
correctness means the same unchecked metadata and the same aliased component
storage as the native call. There is a single operand (no broadcast) and the
factory is autograd-inert (no backward). ``device`` is always passed, because an
omitted ``device`` resolves to CPU and the factory then rejects components that
live elsewhere.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

pytestmark = pytest.mark.underscore_sparse_bsc_tensor_unsafe

# The nine required dtypes, plus the extras the native factory also accepts.
_DTYPES = (
    list(tu.REQUIRED_DTYPES)
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.bool, torch.int16, torch.complex64]
)
_SPECIAL_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]
# Both index dtypes are accepted and stored unchanged.
_INDEX_DTYPES = [torch.int64, torch.int32]
# Pinning needs dense CPU storage, so that negative applies off CPU only.
_ACCELERATOR_BACKEND = torch.device(flag_gems.device).type != "cpu"


def _check_payload_dtype(dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin tensor transfer narrows float64 payloads to float32")


def _cpu_encoded(dtype):
    return flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        torch.float64,
    )


def _make_input(dtype, shape, value_range):
    _check_payload_dtype(dtype)
    if not _cpu_encoded(dtype):
        return tu.make_input(dtype, shape, value_range)
    low, high = (tu.resolve_bound(bound, dtype) for bound in value_range)
    if low == high:
        values = torch.full(shape, low, dtype=torch.float32, device="cpu").to(dtype)
    else:
        values = torch.testing.make_tensor(
            shape, dtype=dtype, device="cpu", low=low, high=high
        )
    result = values.to(flag_gems.device)
    assert result.dtype == dtype
    return result


def _make_special_input(dtype, scenario):
    _check_payload_dtype(dtype)
    if not _cpu_encoded(dtype):
        return tu.make_special_input(dtype, scenario)
    # Encode before transfer: device casts may be absent or narrow float64.
    result = (
        tu.make_special_input(torch.float32, scenario)
        .cpu()
        .to(dtype)
        .to(flag_gems.device)
    )
    assert result.dtype == dtype
    return result


def _component_values(tensor):
    # Observe physical storage first; FP8 device slicing/casts and complex
    # comparison are unavailable on some backends. Retain geometry/lazy flags.
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


def _to_reference(tensor):
    if not utils.TO_CPU:
        return tu.to_reference(tensor)
    # Copy storage, not a device-side FP8/complex view operation. Independence
    # is required even when the source already lives on the CPU.
    values = _component_values(tensor)
    result = torch.empty(0, dtype=values.dtype, device="cpu").set_(
        values.untyped_storage().clone(),
        values.storage_offset(),
        values.size(),
        values.stride(),
    )
    if values.is_conj():
        result = result.conj()
    if values.is_neg():
        result = torch._neg_view(result)
    return result.requires_grad_(tensor.requires_grad)


def _assert_component_equal(result, reference):
    # Candidate device is checked against the original call in _candidate,
    # before CPU observation; this also covers host-only keyword forms.
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape
    assert result.layout == reference.layout == torch.strided
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else result.device)
    tu.assert_result_equal(_component_values(result), _component_values(reference))


def _candidate(*args, **kwargs):
    try:
        device = torch.device(kwargs.get("device", "cpu"))
    except (RuntimeError, TypeError, ValueError):
        # Invalid arguments must still reach the injected implementation.
        flag_gems._sparse_bsc_tensor_unsafe(*args, **kwargs)
        raise AssertionError("candidate accepted an invalid device")
    if device.type != "cpu" and device.index is None:
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    result = flag_gems._sparse_bsc_tensor_unsafe(*args, **kwargs)
    assert result.device == device
    for component in (result.ccol_indices(), result.row_indices(), result.values()):
        assert component.device == device
    return result


def _assert_pinning(result, reference, inputs):
    # Ascend's whole-sparse is_pinned() is unavailable; query each actual
    # dense component. Hygon may accept pin_memory without pinning/copying.
    for produced, expected, source in zip(
        (result.ccol_indices(), result.row_indices(), result.values()),
        (reference.ccol_indices(), reference.row_indices(), reference.values()),
        inputs,
    ):
        assert produced.is_pinned() == expected.is_pinned()
        assert torch._C._is_alias_of(produced, source) == torch._C._is_alias_of(
            expected, source
        )


def _check_accelerator_pin(first, second, values, size):
    kwargs = dict(dtype=values.dtype, device=values.device, pin_memory=True)
    # Keep a separate native copy: a candidate must not mutate its own oracle.
    native_inputs = [t.detach().clone() for t in (first, second, values)]
    try:
        reference = torch.ops.aten._sparse_bsc_tensor_unsafe(
            *native_inputs, size, **kwargs
        )
    except (RuntimeError, TypeError) as error:
        # The candidate call stays outside the native-error catch.
        with pytest.raises(type(error)):
            _candidate(first, second, values, size, **kwargs)
        return
    result = _candidate(first, second, values, size, **kwargs)
    assert result.layout == reference.layout == torch.sparse_bsc
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape
    for produced, expected, source, native_source in zip(
        (result.ccol_indices(), result.row_indices(), result.values()),
        (reference.ccol_indices(), reference.row_indices(), reference.values()),
        (first, second, values),
        native_inputs,
    ):
        assert produced.device == source.device
        assert produced.dtype == expected.dtype
        assert produced.shape == expected.shape
        assert produced.stride() == expected.stride()
        assert produced.storage_offset() == expected.storage_offset()
        assert produced.is_pinned() == expected.is_pinned()
        assert torch._C._is_alias_of(produced, source) == torch._C._is_alias_of(
            expected, native_source
        )
        tu.assert_result_equal(_component_values(produced), _component_values(expected))


def _bsc_structure(size, block, per_column_block, index_dtype=torch.int64):
    """Column-block pointers and block-row indices matching ``size``.

    A BSC tensor keeps two sparse dims, so ``size[0]``/``size[1]`` are the row
    and column extents. A rank < 2 ``size`` has no column extent, so one column
    block is emitted while the requested size is still stored verbatim.
    """
    block_rows, block_cols = block
    nrows = size[0] if size else 1
    ncols = size[1] if len(size) >= 2 else 0
    counts = torch.full(
        (max(ncols // block_cols, 1),), per_column_block, dtype=torch.int64
    )
    offsets = counts.cumsum(0)
    nnz = int(counts.sum())
    ccol = torch.cat([torch.zeros(1, dtype=torch.int64), offsets])
    # Block rows cycle over the row blocks, so every index stays in bounds.
    rows = (torch.arange(nnz) - (offsets - counts).repeat_interleave(counts)) % max(
        nrows // block_rows, 1
    )
    device = flag_gems.device
    return (
        ccol.to(device=device, dtype=index_dtype),
        rows.to(device=device, dtype=index_dtype),
        nnz,
    )


def _bsc_inputs(
    size, dtype, value_range, block=(1, 1), per_column_block=1, index_dtype=torch.int64
):
    """The three components plus the ``size`` list of one BSC descriptor."""
    ccol, rows, nnz = _bsc_structure(size, block, per_column_block, index_dtype)
    values = _make_input(dtype, (nnz,) + tuple(block) + tuple(size[2:]), value_range)
    return ccol, rows, values, list(size)


def _assert_matches(res_out, ref_out, size, dtype):
    """Layout, unchecked metadata and stored components match the reference."""
    assert res_out.layout == ref_out.layout == torch.sparse_bsc
    assert res_out.dtype == dtype
    assert tuple(res_out.shape) == tuple(size)
    assert res_out.sparse_dim() == ref_out.sparse_dim() == 2
    assert res_out.dense_dim() == ref_out.dense_dim()
    assert res_out.ccol_indices().dtype == ref_out.ccol_indices().dtype
    assert res_out.row_indices().dtype == ref_out.row_indices().dtype
    _assert_component_equal(res_out.ccol_indices(), ref_out.ccol_indices())
    _assert_component_equal(res_out.row_indices(), ref_out.row_indices())
    _assert_component_equal(res_out.values(), ref_out.values())


def _assert_payload_is_aliased(res_out, ccol, rows, values):
    """Each component is installed by storage alias with its metadata intact.

    The shared assertions cover dtype and shape of the outer tensor only, so
    stride, storage offset and storage sharing are asserted here.
    """
    assert res_out.device == ccol.device
    for stored, given in (
        (res_out.ccol_indices(), ccol),
        (res_out.row_indices(), rows),
        (res_out.values(), values),
    ):
        assert torch._C._is_alias_of(stored, given)
        assert stored.data_ptr() == given.data_ptr()
        assert stored.shape == given.shape
        assert stored.stride() == given.stride()
        assert stored.storage_offset() == given.storage_offset()


@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test__sparse_bsc_tensor_unsafe(shape, value_range, dtype):
    ccol, rows, values, size = _bsc_inputs(tuple(shape), dtype, value_range)

    # ``dtype`` is requested explicitly because the factory's own default is the
    # global default dtype, which the values dtype must equal. ``layout`` and
    # ``pin_memory`` stay omitted so their defaults are exercised.
    ref_ccol, ref_rows, ref_values = (_to_reference(t) for t in (ccol, rows, values))
    ref_out = torch.ops.aten._sparse_bsc_tensor_unsafe(
        ref_ccol, ref_rows, ref_values, size, dtype=dtype, device=ref_ccol.device
    )
    res_out = _candidate(ccol, rows, values, size, dtype=dtype, device=flag_gems.device)

    _assert_matches(res_out, ref_out, size, dtype)
    _assert_payload_is_aliased(res_out, ccol, rows, values)


def test__sparse_bsc_tensor_unsafe_default_keywords():
    """The optional keywords are omitted: dtype/layout fall back to defaults."""
    ccol, rows, values, size = _bsc_inputs((4, 6), torch.float32, ["-1", "1"], (1, 2))

    ref_ccol, ref_rows, ref_values = (_to_reference(t) for t in (ccol, rows, values))
    ref_out = torch.ops.aten._sparse_bsc_tensor_unsafe(
        ref_ccol, ref_rows, ref_values, size, device=ref_ccol.device
    )
    res_out = _candidate(ccol, rows, values, size, device=flag_gems.device)

    _assert_matches(res_out, ref_out, size, torch.float32)
    assert res_out.layout == torch.sparse_bsc


# Distinct call forms on tiny descriptors: index dtype, block size, dense tail,
# empty payload and rank < 2 size. Shrinking cannot change what they exercise,
# so they stay in --quick too.
_DESCRIPTOR_CASES = [
    ("single_block", (4, 3), (1, 1), 1),
    ("block_2x2", (4, 4), (2, 2), 1),
    ("block_1x2", (4, 6), (1, 2), 1),
    ("stacked_blocks", (6, 4), (1, 1), 2),
    ("empty_payload", (4, 3), (1, 1), 0),
    ("dense_tail", (4, 6, 4), (1, 1), 1),
    ("dense_tail_2d", (4, 6, 2, 3), (1, 2), 1),
    ("zero_dim_size", (), (1, 1), 1),
    ("one_dim_size", (256,), (1, 1), 1),
]


@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize(
    "size,block,per_column_block",
    [case[1:] for case in _DESCRIPTOR_CASES],
    ids=[case[0] for case in _DESCRIPTOR_CASES],
)
def test__sparse_bsc_tensor_unsafe_descriptor(
    size, block, per_column_block, index_dtype
):
    ccol, rows, values, full_size = _bsc_inputs(
        size, torch.float32, ["-1", "1"], block, per_column_block, index_dtype
    )

    ref_ccol, ref_rows, ref_values = (_to_reference(t) for t in (ccol, rows, values))
    ref_out = torch.ops.aten._sparse_bsc_tensor_unsafe(
        ref_ccol,
        ref_rows,
        ref_values,
        full_size,
        dtype=torch.float32,
        device=ref_ccol.device,
    )
    res_out = _candidate(
        ccol, rows, values, full_size, dtype=torch.float32, device=flag_gems.device
    )

    _assert_matches(res_out, ref_out, full_size, torch.float32)
    assert res_out.ccol_indices().dtype == index_dtype
    assert res_out.row_indices().dtype == index_dtype
    _assert_payload_is_aliased(res_out, ccol, rows, values)


@pytest.mark.parametrize(
    "pointer_delta", [-1, 1], ids=["truncated_pointers", "oversized_pointers"]
)
def test__sparse_bsc_tensor_unsafe_unchecked_pointers(pointer_delta):
    """Pointers that do not match the column extent are stored verbatim.

    A BSC tensor needs ``ncols + 1`` pointers, but the factory checks no length,
    so the candidate has to keep the same unchecked contract instead of
    validating or densifying the descriptor.
    """
    size = (4, 6)
    base_ccol, rows, _ = _bsc_structure(size, (1, 1), 1)
    extended = torch.cat([base_ccol, base_ccol[-1:]])
    ccol = extended[: len(base_ccol) + pointer_delta]
    values = _make_input(torch.float32, (6, 1, 1), ["-1", "1"])

    ref_ccol, ref_rows, ref_values = (_to_reference(t) for t in (ccol, rows, values))
    ref_out = torch.ops.aten._sparse_bsc_tensor_unsafe(
        ref_ccol,
        ref_rows,
        ref_values,
        list(size),
        dtype=torch.float32,
        device=ref_ccol.device,
    )
    res_out = _candidate(
        ccol, rows, values, list(size), dtype=torch.float32, device=flag_gems.device
    )

    _assert_matches(res_out, ref_out, size, torch.float32)
    assert res_out.ccol_indices().numel() == ccol.numel()
    _assert_payload_is_aliased(res_out, ccol, rows, values)


def test__sparse_bsc_tensor_unsafe_keeps_component_metadata():
    """Strided components and a sliced pointer tensor keep their metadata."""
    size = (4, 6)
    _, rows, nnz = _bsc_structure(size, (1, 2), 1)
    # Pointers sliced out of a larger storage: the offset must survive and the
    # leading element must not leak into the stored component.
    ccol = torch.tensor([4096, 0, 1, 2, 3], dtype=torch.int64, device=flag_gems.device)[
        1:
    ]
    rows = torch.arange(6, device=flag_gems.device)[::2]
    values = _make_input(torch.float32, (2, 1, nnz), ["-1", "1"]).transpose(0, 2)
    assert not values.is_contiguous()

    ref_ccol, ref_rows, ref_values = (_to_reference(t) for t in (ccol, rows, values))
    ref_out = torch.ops.aten._sparse_bsc_tensor_unsafe(
        ref_ccol,
        ref_rows,
        ref_values,
        list(size),
        dtype=torch.float32,
        device=ref_ccol.device,
    )
    res_out = _candidate(
        ccol, rows, values, list(size), dtype=torch.float32, device=flag_gems.device
    )

    _assert_matches(res_out, ref_out, size, torch.float32)
    assert res_out.ccol_indices().storage_offset() == 1
    assert res_out.ccol_indices().tolist() == [0, 1, 2, 3]
    assert res_out.values().stride() == values.stride()
    assert not res_out.values().is_contiguous()
    _assert_payload_is_aliased(res_out, ccol, rows, values)


def test__sparse_bsc_tensor_unsafe_reuses_component_storage():
    """Writes through the given tensors are visible in the sparse tensor."""
    ccol, rows, values, size = _bsc_inputs((4, 4), torch.float32, ["-1", "1"])

    ref_ccol, ref_rows, ref_values = (_to_reference(t) for t in (ccol, rows, values))
    ref_out = torch.ops.aten._sparse_bsc_tensor_unsafe(
        ref_ccol,
        ref_rows,
        ref_values,
        size,
        dtype=torch.float32,
        device=ref_ccol.device,
    )
    res_out = _candidate(
        ccol, rows, values, size, dtype=torch.float32, device=flag_gems.device
    )
    _assert_matches(res_out, ref_out, size, torch.float32)

    values.fill_(3.25)
    assert torch.equal(res_out.values(), torch.full_like(values, 3.25))
    res_out.values().fill_(-1.5)
    assert torch.equal(values, torch.full_like(values, -1.5))
    ccol.fill_(0)
    assert torch.equal(res_out.ccol_indices(), torch.zeros_like(ccol))


def test__sparse_bsc_tensor_unsafe_optional_kwargs():
    """Every optional keyword is passed explicitly; the grid omits them all."""
    ccol, rows, values, size = _bsc_inputs((4, 6), torch.float32, ["-1", "1"], (1, 2))

    ref_ccol, ref_rows, ref_values = (_to_reference(t) for t in (ccol, rows, values))
    ref_out = torch.ops.aten._sparse_bsc_tensor_unsafe(
        ref_ccol,
        ref_rows,
        ref_values,
        size,
        dtype=torch.float32,
        layout=torch.sparse_bsc,
        device=ref_ccol.device,
    )
    res_out = _candidate(
        ccol,
        rows,
        values,
        size,
        dtype=torch.float32,
        layout=torch.sparse_bsc,
        device=flag_gems.device,
    )

    _assert_matches(res_out, ref_out, size, torch.float32)


def test__sparse_bsc_tensor_unsafe_cpu_payload_and_pin_memory():
    """CPU payloads with ``pin_memory`` - the only valid use of that keyword."""
    ccol, rows, values, size = _bsc_inputs((4, 6), torch.float32, ["-1", "1"], (1, 2))
    ccol, rows, values = ccol.cpu(), rows.cpu(), values.cpu()
    device = torch.device("cpu")

    ref_out = torch.ops.aten._sparse_bsc_tensor_unsafe(
        ccol,
        rows,
        values,
        size,
        dtype=torch.float32,
        layout=torch.sparse_bsc,
        device=device,
        pin_memory=True,
    )
    res_out = _candidate(
        ccol,
        rows,
        values,
        size,
        dtype=torch.float32,
        layout=torch.sparse_bsc,
        device=device,
        pin_memory=True,
    )

    assert res_out.device == device
    # Pinning copies the components into pinned storage, so this call shares no
    # storage with the inputs and only the stored values are compared.
    _assert_pinning(res_out, ref_out, (ccol, rows, values))
    _assert_matches(res_out, ref_out, size, torch.float32)


@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test__sparse_bsc_tensor_unsafe_special_values(dtype, scenario):
    # A (3, 5) size gives five column blocks, one per special payload entry, so
    # the stored values keep the whole nan/inf payload of that scenario.
    size = (3, 5)
    ccol, rows, _ = _bsc_structure(size, (1, 1), 1)
    values = _make_special_input(dtype, scenario)
    if _cpu_encoded(dtype):
        values = _component_values(values).reshape(5, 1, 1).to(flag_gems.device)
    else:
        values = values.reshape(5, 1, 1)

    ref_ccol, ref_rows, ref_values = (_to_reference(t) for t in (ccol, rows, values))
    ref_out = torch.ops.aten._sparse_bsc_tensor_unsafe(
        ref_ccol,
        ref_rows,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_ccol.device,
    )
    res_out = _candidate(
        ccol, rows, values, list(size), dtype=dtype, device=flag_gems.device
    )

    _assert_matches(res_out, ref_out, size, dtype)
    _assert_payload_is_aliased(res_out, ccol, rows, values)


def test__sparse_bsc_tensor_unsafe_negative_missing_size():
    ccol, rows, values, _ = _bsc_inputs((4, 4), torch.float32, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        _candidate(ccol, rows, values, device=flag_gems.device)


def test__sparse_bsc_tensor_unsafe_negative_non_tensor_pointer():
    _, rows, values, size = _bsc_inputs((4, 4), torch.float32, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        _candidate([0, 2, 4, 6], rows, values, size, device=flag_gems.device)


def test__sparse_bsc_tensor_unsafe_negative_negative_extent():
    ccol, rows, values, _ = _bsc_inputs((4, 4), torch.float32, ["-1", "1"])
    with pytest.raises(RuntimeError):
        _candidate(ccol, rows, values, [-4, 4], device=flag_gems.device)


def test__sparse_bsc_tensor_unsafe_negative_dtype_mismatch():
    ccol, rows, values, size = _bsc_inputs((4, 4), torch.float32, ["-1", "1"])
    with pytest.raises(RuntimeError):
        _candidate(ccol, rows, values, size, dtype=torch.int32, device=flag_gems.device)


def test__sparse_bsc_tensor_unsafe_negative_wrong_layout():
    ccol, rows, values, size = _bsc_inputs((4, 4), torch.float32, ["-1", "1"])
    with pytest.raises(RuntimeError):
        _candidate(
            ccol, rows, values, size, layout=torch.sparse_coo, device=flag_gems.device
        )


@pytest.mark.skipif(
    not _ACCELERATOR_BACKEND, reason="CPU components already match a CPU request"
)
def test__sparse_bsc_tensor_unsafe_negative_device_mismatch():
    ccol, rows, values, size = _bsc_inputs((4, 4), torch.float32, ["-1", "1"])
    with pytest.raises(RuntimeError):
        _candidate(ccol, rows, values, size, device=torch.device("cpu"))


@pytest.mark.skipif(
    not _ACCELERATOR_BACKEND, reason="only dense CPU storage can be pinned"
)
def test__sparse_bsc_tensor_unsafe_negative_pin_accelerator_payload():
    ccol, rows, values, size = _bsc_inputs((4, 4), torch.float32, ["-1", "1"])
    _check_accelerator_pin(ccol, rows, values, size)


@pytest.mark.underscore_sparse_bsc_tensor_unsafe
@pytest.mark.parametrize(
    "dtype", [dtype for dtype in _DTYPES if dtype.is_floating_point or dtype.is_complex]
)
def test__sparse_bsc_tensor_unsafe_requires_grad_input(dtype):
    # Unsafe factories alias the payload but do not create an autograd edge.
    compressed = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    plain = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    values_shape = (2, 2, 2)
    size = [4, 4]
    values = _make_input(dtype, values_shape, ["-1", "1"]).requires_grad_()
    ref_first, ref_second, ref_values = (
        _to_reference(t) for t in (compressed, plain, values)
    )
    ref_out = torch.ops.aten._sparse_bsc_tensor_unsafe(
        ref_first,
        ref_second,
        ref_values,
        size,
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=ref_values.device,
    )
    res_out = _candidate(
        compressed,
        plain,
        values,
        size,
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=values.device,
    )

    assert not res_out.requires_grad
    assert res_out.requires_grad == ref_out.requires_grad
    assert res_out.grad_fn is ref_out.grad_fn is None
    assert res_out.values().data_ptr() == values.data_ptr()
    _assert_component_equal(res_out.values(), ref_out.values())
