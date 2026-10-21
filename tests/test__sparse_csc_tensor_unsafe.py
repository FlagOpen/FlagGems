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

# _sparse_csc_tensor_unsafe is the unchecked CSC constructor: it validates
# neither the metadata nor the buffer lengths and stores the caller's three
# tensors by alias. The tests assert the reported metadata (layout, size,
# sparse_dim, dense_dim, nnz, component shape/dtype) and the alias contract,
# and compare the stored index/value buffers through the shared helpers.
# Malformed metadata is compared verbatim and never densified. The constructor
# has no broadcast operands and no kernel backward, so those dimensions do not
# apply here.

_INDEX_DTYPES = [torch.int32, torch.int64]

_VALUE_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
]

_SPEC_SHAPES = [
    (),
    (1,),
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]
_QUICK_SPEC_SHAPES = [(2, 19, 7)]

_SPECIAL_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
]


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
        flag_gems._sparse_csc_tensor_unsafe(*args, **kwargs)
        raise AssertionError("candidate accepted an invalid device")
    if device.type != "cpu" and device.index is None:
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    result = flag_gems._sparse_csc_tensor_unsafe(*args, **kwargs)
    assert result.device == device
    for component in (result.ccol_indices(), result.row_indices(), result.values()):
        assert component.device == device
    return result


def _check_accelerator_pin(first, second, values, size):
    kwargs = dict(dtype=values.dtype, device=values.device, pin_memory=True)
    # Keep a separate native copy: a candidate must not mutate its own oracle.
    native_inputs = [t.detach().clone() for t in (first, second, values)]
    try:
        reference = torch.ops.aten._sparse_csc_tensor_unsafe(
            *native_inputs, size, **kwargs
        )
    except (RuntimeError, TypeError) as error:
        # The candidate call stays outside the native-error catch.
        with pytest.raises(type(error)):
            _candidate(first, second, values, size, **kwargs)
        return
    result = _candidate(first, second, values, size, **kwargs)
    assert result.layout == reference.layout == torch.sparse_csc
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


def _matrix_size(spec_shape):
    # CSC needs the trailing (rows, cols) axes, so the spec's rank-0 and rank-1
    # shapes widen to the smallest valid matrix. The bare rank-0/rank-1 `size`
    # lists with empty components are exercised by the unchecked cases below.
    if len(spec_shape) >= 2:
        return [int(dim) for dim in spec_shape]
    if len(spec_shape) == 1:
        return [max(int(spec_shape[0]), 1), 4]
    return [1, 1]


def _csc_indices(size, per_column, index_dtype=torch.int64, device=None):
    # Compressed column pointers plus sorted, duplicate-free row indices.
    device = flag_gems.device if device is None else device
    batch, rows, cols = list(size[:-2]), int(size[-2]), int(size[-1])
    per = max(0, min(int(per_column), rows))
    ccol = torch.arange(cols + 1, dtype=index_dtype, device=device) * per
    row = torch.arange(per, dtype=index_dtype, device=device).repeat(cols)
    nnz = per * cols
    if batch:
        ccol = ccol.expand(*batch, cols + 1).contiguous()
        row = row.expand(*batch, nnz).contiguous()
    return ccol, row, batch + [nnz]


def _assert_metadata(res_out, ref_out, size, values_device):
    assert res_out.layout == torch.sparse_csc
    assert res_out.dtype == ref_out.dtype
    assert tuple(res_out.shape) == tuple(ref_out.shape) == tuple(size)
    assert res_out.sparse_dim() == ref_out.sparse_dim()
    assert res_out.dense_dim() == ref_out.dense_dim()
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out)
    # The output device follows the requested/input device, not the reference's.
    assert res_out.device == values_device


def _assert_components(res_out, ref_out, index_dtype):
    # Sparse CSC tensors expose no stride() (native raises 'Sparse CSC tensors
    # do not have strides'); the aliased component views are compared instead.
    assert res_out.ccol_indices().dtype == index_dtype
    assert res_out.row_indices().dtype == index_dtype
    assert res_out.ccol_indices().shape == ref_out.ccol_indices().shape
    assert res_out.row_indices().shape == ref_out.row_indices().shape
    assert res_out.values().shape == ref_out.values().shape
    assert res_out.values().dtype == ref_out.values().dtype
    _assert_component_equal(res_out.ccol_indices(), ref_out.ccol_indices())
    _assert_component_equal(res_out.row_indices(), ref_out.row_indices())
    _assert_component_equal(res_out.values(), ref_out.values())


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _VALUE_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "spec_shape", tu.selected_cases(_SPEC_SHAPES, quick=_QUICK_SPEC_SHAPES)
)
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test__sparse_csc_tensor_unsafe_values(spec_shape, value_range, dtype):
    size = _matrix_size(spec_shape)
    ccol, row, values_shape = _csc_indices(size, 1)
    values = _make_input(dtype, values_shape, value_range)

    ref_ccol = _to_reference(ccol)
    ref_row = _to_reference(row)
    ref_values = _to_reference(values)
    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        ref_ccol,
        ref_row,
        ref_values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _candidate(
        ccol,
        row,
        values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, torch.int64)
    # The constructor stores the buffers it is given; they must stay unchanged.
    _assert_component_equal(ccol, ref_ccol)
    _assert_component_equal(row, ref_row)
    _assert_component_equal(values, ref_values)


# (size, entries per column): the second value shapes the compressed structure
# and the dense prefixes exercise the batched sizes.
_STRUCT_CASES = [
    ([3, 2], 1),
    ([5, 7], 3),
    ([1, 256], 1),
    ([2, 3, 4], 1),
    ([2, 2, 3, 4], 2),
    ([16, 7, 57, 32, 29], 1),
]
# Quick keeps every cheap branch (both index dtypes, single-column and batched
# sizes, multi-entry columns); only the large 5-D size stays default-only.
_QUICK_STRUCT_CASES = [
    ([3, 2], 1),
    ([5, 7], 3),
    ([1, 256], 1),
    ([2, 3, 4], 1),
    ([2, 2, 3, 4], 2),
]


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "size, per_column", tu.selected_cases(_STRUCT_CASES, quick=_QUICK_STRUCT_CASES)
)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test__sparse_csc_tensor_unsafe_structure(size, per_column, dtype, index_dtype):
    ccol, row, values_shape = _csc_indices(size, per_column, index_dtype=index_dtype)
    values = _make_input(dtype, values_shape, ["-1", "1"])

    ref_ccol = _to_reference(ccol)
    ref_row = _to_reference(row)
    ref_values = _to_reference(values)
    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        ref_ccol,
        ref_row,
        ref_values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _candidate(
        ccol,
        row,
        values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, index_dtype)


# Native-accepted malformed metadata (probed on CPU and CUDA): the unchecked
# constructor reports exactly what it is given. A rank-0 ccol or a 2-D values
# tensor raises dense_dim() to 1, and the buffer lengths are not validated.
# A rank-0/rank-1 size keeps the requested shape with empty components.
# (case, size, ccol, row, values shape)
_UNCHECKED_CASES = [
    ("values-longer-than-nnz", [3, 2], [0, 1, 2], [0, 1], [5]),
    ("values-shorter-than-nnz", [3, 2], [0, 1, 2], [0, 1], [1]),
    ("duplicate-rows", [2, 2], [0, 1, 3], [0, 0, 0], [3]),
    ("unsorted-rows", [3, 1], [0, 3, 3], [1, 0, 2], [3]),
    ("ccol-past-nnz", [3, 2], [0, 1, 9], [0, 1], [2]),
    ("short-ccol", [3, 2], [0, 1], [0, 1], [2]),
    ("rank0-ccol", [3, 2], 2, [0, 1], [2]),
    ("size-rank-mismatch", [3, 3, 3], [0, 1, 2], [0, 1], [2]),
    ("size-rank0", [], [0, 1, 2], [0, 1], [2]),
    ("zero-columns", [3, 0], [0, 1, 2], [0, 1], [2]),
    ("zero-rows", [0, 0], [0, 1, 2], [0, 1], [2]),
    ("dense-tail", [3, 2], [0, 2], [0, 1], [2, 1]),
    ("rank1-empty-size", [1073741824], [], [], [0]),
    ("rank0-empty-size", [], [], [], [0]),
]


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "case, size, ccol_spec, row_spec, values_shape", _UNCHECKED_CASES
)
def test__sparse_csc_tensor_unsafe_unchecked_metadata(
    case, size, ccol_spec, row_spec, values_shape
):
    ccol = torch.tensor(ccol_spec, dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor(row_spec, dtype=torch.int64, device=flag_gems.device)
    values = _make_input(torch.float32, values_shape, ["-1", "1"])

    ref_first, ref_second, ref_values = (_to_reference(t) for t in (ccol, row, values))
    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        ref_first,
        ref_second,
        ref_values,
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_csc,
        device=ref_values.device,
    )
    res_out = _candidate(
        ccol,
        row,
        values,
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, torch.int64)


@pytest.mark.underscore_sparse_csc_tensor_unsafe
def test__sparse_csc_tensor_unsafe_aliases_component_storage():
    # Strided views with non-zero storage offsets: the constructor keeps the
    # buffers themselves, so the components must expose the same storage, shape,
    # stride and offset instead of a normalized copy.
    ccol_base = torch.zeros(9, dtype=torch.int64, device=flag_gems.device)
    ccol_base[::2] = torch.arange(5, dtype=torch.int64, device=flag_gems.device)
    ccol = ccol_base[::2]
    row_base = torch.zeros(9, dtype=torch.int64, device=flag_gems.device)
    row_base[1:9:2] = torch.arange(4, dtype=torch.int64, device=flag_gems.device)
    row = row_base[1:9:2]
    values_base = torch.zeros(9, dtype=torch.float32, device=flag_gems.device)
    values_base[1:9:2] = torch.arange(
        1, 5, dtype=torch.float32, device=flag_gems.device
    )
    values = values_base[1:9:2]
    size = [6, 4]

    ref_first, ref_second, ref_values = (_to_reference(t) for t in (ccol, row, values))
    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        ref_first,
        ref_second,
        ref_values,
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_csc,
        device=ref_values.device,
    )
    res_out = _candidate(
        ccol,
        row,
        values,
        list(size),
        dtype=torch.float32,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, torch.int64)

    for produced, source in (
        (res_out.ccol_indices(), ccol),
        (res_out.row_indices(), row),
        (res_out.values(), values),
    ):
        assert (
            produced.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        )
        assert produced.data_ptr() == source.data_ptr()
        assert produced.shape == source.shape
        assert produced.stride() == source.stride()
        assert produced.storage_offset() == source.storage_offset()

    res_out.values().fill_(7.5)
    assert torch.equal(values, torch.full_like(values, 7.5))


# The schema defaults are dtype=float32, layout=sparse_csc, device=cpu and
# pin_memory; the default device is CPU, so the omitted-argument forms use CPU
# buffers with the reference on the same CPU contract.
_OPTION_CASES = [
    (
        "explicit-kwargs",
        {"dtype": torch.float32, "layout": torch.sparse_csc, "device": "cpu"},
    ),
    ("omit-layout", {"dtype": torch.float32, "device": "cpu"}),
    ("omit-device", {"dtype": torch.float32, "layout": torch.sparse_csc}),
    ("omit-dtype", {"layout": torch.sparse_csc, "device": "cpu"}),
    (
        "pin-memory-false",
        {
            "dtype": torch.float32,
            "layout": torch.sparse_csc,
            "device": "cpu",
            "pin_memory": False,
        },
    ),
    (
        "pin-memory-true",
        {
            "dtype": torch.float32,
            "layout": torch.sparse_csc,
            "device": "cpu",
            "pin_memory": True,
        },
    ),
]


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.parametrize("case, kwargs", _OPTION_CASES)
def test__sparse_csc_tensor_unsafe_optional_kwargs(case, kwargs):
    ccol = torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu")
    row = torch.tensor([0, 1], dtype=torch.int64, device="cpu")
    values = torch.tensor([1.5, -2.25], dtype=torch.float32, device="cpu")
    size = [3, 2]

    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        _to_reference(ccol),
        _to_reference(row),
        _to_reference(values),
        list(size),
        **kwargs,
    )
    res_out = _candidate(ccol, row, values, list(size), **kwargs)

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, torch.int64)
    assert res_out.values().is_pinned() == ref_out.values().is_pinned()


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "dtype, scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test__sparse_csc_tensor_unsafe_special_values(dtype, scenario):
    values = _make_special_input(dtype, scenario)
    ccol = torch.tensor([0, 2, 5], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    size = [3, 2]

    ref_first, ref_second, ref_values = (_to_reference(t) for t in (ccol, row, values))
    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        ref_first,
        ref_second,
        ref_values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_values.device,
    )
    res_out = _candidate(
        ccol,
        row,
        values,
        list(size),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=flag_gems.device,
    )

    _assert_metadata(res_out, ref_out, size, values.device)
    _assert_components(res_out, ref_out, torch.int64)


def _basic_buffers(dtype=torch.float32):
    device = flag_gems.device
    ccol = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    row = torch.tensor([0, 1], dtype=torch.int64, device=device)
    _check_payload_dtype(dtype)
    values = torch.tensor([1.5, -2.25], dtype=dtype, device="cpu").to(device)
    assert values.dtype == dtype
    return ccol, row, values


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "values_dtype, declared_dtype",
    [(torch.float16, torch.float32), (torch.float64, torch.float32)],
)
def test__sparse_csc_tensor_unsafe_rejects_dtype_mismatch(values_dtype, declared_dtype):
    ccol, row, values = _basic_buffers(values_dtype)
    with pytest.raises((RuntimeError, TypeError)):
        _candidate(
            ccol,
            row,
            values,
            [3, 2],
            dtype=declared_dtype,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.underscore_sparse_csc_tensor_unsafe
def test__sparse_csc_tensor_unsafe_rejects_missing_dtype_for_other_values():
    ccol, row, values = _basic_buffers(torch.float64)
    with pytest.raises((RuntimeError, TypeError)):
        _candidate(
            ccol,
            row,
            values,
            [3, 2],
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.parametrize("layout", [torch.sparse_csr, torch.sparse_coo])
def test__sparse_csc_tensor_unsafe_rejects_foreign_layout(layout):
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError, ValueError, NotImplementedError)):
        _candidate(
            ccol,
            row,
            values,
            [3, 2],
            dtype=torch.float32,
            layout=layout,
            device=flag_gems.device,
        )


@pytest.mark.underscore_sparse_csc_tensor_unsafe
def test__sparse_csc_tensor_unsafe_rejects_non_list_size():
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _candidate(
            ccol,
            row,
            values,
            3,
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.underscore_sparse_csc_tensor_unsafe
def test__sparse_csc_tensor_unsafe_rejects_unknown_device():
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        _candidate(
            ccol,
            row,
            values,
            [3, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device="not-a-device",
        )


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.skipif(flag_gems.device == "cpu", reason="needs a non-CPU device")
def test__sparse_csc_tensor_unsafe_rejects_missing_device_on_accelerator():
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError)):
        _candidate(
            ccol,
            row,
            values,
            [3, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
        )


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.skipif(flag_gems.device == "cpu", reason="needs a non-CPU device")
def test__sparse_csc_tensor_unsafe_rejects_cross_device_buffers():
    ccol, row, values = _basic_buffers()
    with pytest.raises((RuntimeError, TypeError)):
        _candidate(
            ccol,
            row,
            values.to("cpu"),
            [3, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.skipif(flag_gems.device == "cpu", reason="needs a non-CPU device")
def test__sparse_csc_tensor_unsafe_rejects_pinned_accelerator_buffers():
    ccol, row, values = _basic_buffers()
    _check_accelerator_pin(ccol, row, values, [3, 2])


@pytest.mark.underscore_sparse_csc_tensor_unsafe
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in _VALUE_DTYPES if dtype.is_floating_point or dtype.is_complex],
)
def test__sparse_csc_tensor_unsafe_requires_grad_input(dtype):
    # Unsafe factories alias the payload but do not create an autograd edge.
    compressed = torch.tensor([0, 1, 2], dtype=torch.int64, device=flag_gems.device)
    plain = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
    values_shape = (2,)
    size = [2, 2]
    values = _make_input(dtype, values_shape, ["-1", "1"]).requires_grad_()
    ref_first, ref_second, ref_values = (
        _to_reference(t) for t in (compressed, plain, values)
    )
    ref_out = torch.ops.aten._sparse_csc_tensor_unsafe(
        ref_first,
        ref_second,
        ref_values,
        size,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_values.device,
    )
    res_out = _candidate(
        compressed,
        plain,
        values,
        size,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=values.device,
    )

    assert not res_out.requires_grad
    assert res_out.requires_grad == ref_out.requires_grad
    assert res_out.grad_fn is ref_out.grad_fn is None
    assert res_out.values().data_ptr() == values.data_ptr()
    _assert_component_equal(res_out.values(), ref_out.values())
