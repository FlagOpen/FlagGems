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

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Empty FP8 casts work; constant fills fail even for empty values.


# Store CSC indices and values verbatim, including repeated/unsorted rows.
# Pass dtype and device explicitly to the ATen factory.
_EXACT_CSC_DTYPES = [torch.int8, torch.uint8] + utils.ALL_INT_DTYPES + [torch.bool]
_FLOAT_STORAGE_DTYPES = utils.ALL_FLOAT_DTYPES + [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
_CSC_DTYPES = _FLOAT_STORAGE_DTYPES + _EXACT_CSC_DTYPES
_INDEX_DTYPES = [torch.int32, torch.int64]

# (matrix_shape, nnz), then batched variants.
_CSC_CASES = [
    ((1, 8), 4),
    ((8, 1), 4),
    ((4, 4), 16),
    ((5, 7), 13),
    ((7, 5), 13),
    ((4, 4), 0),
]
_CSC_BATCHED_CASES = [
    ((2, 4, 4), 4),
    ((3, 6, 5), 6),
]

# (ccol_indices, row_indices); size is inferred by the factory.
_CSC_NO_SIZE_CASES = [
    ([0, 2, 3, 5], [0, 2, 1, 2, 3]),
    ([0, 3, 3, 4], [0, 1, 2, 3]),
    ([0, 0, 0], []),
    ([0, 2, 4], [0, 1, 0, 1]),
]

# Skip scalars; map 1-D shapes to square matrices.
_CSC_SHAPE_CASES = [
    (
        (shape + shape, shape[0])
        if len(shape) == 1
        else (shape, min(shape[-2] * shape[-1], 8))
    )
    for shape in tu.selected_shapes()
    if shape
]
_BOUNDARY_RANGES = [
    ["min", "min"],
    ["max", "max"],
    ["0", "0"],
    ["1", "1"],
    ["-1", "-1"],
]


def _required_devices():
    target = torch.empty(0, dtype=torch.uint8, device=flag_gems.device).device
    return target, torch.device("cpu") if utils.TO_CPU else target


def _storage_bytes(inp):
    storage = inp.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _copy_tensor(inp, device):
    # Copy physical bytes before reinterpreting, avoiding typed Double narrowing
    # and unsupported device FP8 casts. Reconstruct the original view geometry.
    raw = _storage_bytes(inp).to(device=device).clone()
    result = raw.view(inp.dtype).as_strided(
        inp.shape, inp.stride(), inp.storage_offset()
    )
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    result.requires_grad_(inp.requires_grad)
    return result


def _check_dense(inp, dtype, shape, device):
    assert isinstance(inp, torch.Tensor) and inp.layout == torch.strided
    assert inp.dtype == dtype and tuple(inp.shape) == tuple(shape)
    assert inp.device == device
    return inp


def _make_values(nnz, dtype, shape=None, value_range=("-1", "1")):
    shape = (nnz,) if shape is None else tuple(shape)
    required_device = _required_devices()[0]
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        *_FP8_DTYPES,
        torch.float64,
    ):
        # Original TU bounds/full/make_tensor recipe, encoded in the requested
        # CPU dtype before byte transport; this also serves the Double negative.
        low = tu.resolve_bound(value_range[0], dtype)
        high = tu.resolve_bound(value_range[1], dtype)
        if low == high:
            host = torch.full(shape, low, dtype=dtype, device="cpu")
        else:
            host = torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        values = _copy_tensor(host, required_device)
    elif dtype == torch.uint8:
        low = max(int(tu.resolve_bound(value_range[0], dtype)), 0)
        high = max(int(tu.resolve_bound(value_range[1], dtype)), 0)
        if low > high:
            low = high
        if low == high:
            values = torch.full(shape, low, device=flag_gems.device, dtype=dtype)
        else:
            values = torch.testing.make_tensor(
                shape, dtype=dtype, device=flag_gems.device, low=low, high=high
            )
    else:
        values = tu.make_input(dtype, shape, value_range).to(required_device)
    return _check_dense(values, dtype, shape, required_device)


def _make_special_values(dtype, scenario, *, required_device):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        *_FP8_DTYPES,
        torch.float64,
    ):
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        host = torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu").to(
            dtype
        )
        values = _copy_tensor(host, required_device)
    else:
        values = tu.make_special_input(dtype, scenario)
    return _check_dense(values, dtype, (5,), required_device)


def _make_csc_inputs(
    shape, nnz, dtype, index_dtype=torch.int64, value_range=("-1", "1")
):
    device = _required_devices()[0]
    batch, (M, N) = shape[:-2], shape[-2:]
    assert 0 <= nnz <= M * N
    counts = torch.full((N,), nnz // N, dtype=torch.long)
    counts[: nnz % N] += 1
    ccol = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    row = torch.arange(nnz) - torch.repeat_interleave(ccol[:-1], counts)
    ccol = ccol.expand(batch + (N + 1,)).contiguous()
    row = row.expand(batch + (nnz,)).contiguous()
    values = _make_values(nnz, dtype, shape=batch + (nnz,), value_range=value_range)
    ccol = ccol.to(device=device, dtype=index_dtype)
    row = row.to(device=device, dtype=index_dtype)
    _check_dense(ccol, index_dtype, batch + (N + 1,), device)
    _check_dense(row, index_dtype, batch + (nnz,), device)
    _check_dense(values, dtype, batch + (nnz,), device)
    return ccol, row, values


def _to_reference(inp, *, required_device, reference_device):
    assert isinstance(inp, torch.Tensor) and inp.device == required_device
    result = _copy_tensor(inp, reference_device)
    _check_dense(result, inp.dtype, inp.shape, reference_device)
    assert (
        result.stride() == inp.stride()
        and result.storage_offset() == inp.storage_offset()
    )
    assert result.is_conj() == inp.is_conj() and result.is_neg() == inp.is_neg()
    assert result.untyped_storage()._cdata != inp.untyped_storage()._cdata
    return result


def _input_state(inp):
    assert isinstance(inp, torch.Tensor) and inp.layout == torch.strided
    storage = inp.untyped_storage()
    signature = (
        type(inp),
        inp.device,
        inp.dtype,
        inp.layout,
        tuple(inp.shape),
        tuple(inp.stride()),
        inp.storage_offset(),
        inp.requires_grad,
        inp._version,
        inp.is_conj(),
        inp.is_neg(),
        id(inp._base),
        storage._cdata,
        inp.data_ptr(),
        storage.data_ptr(),
        storage.nbytes(),
    )
    return signature, storage, inp._base


def _call_candidate(
    ccol, row, values, *size, required_device, expected_shape, **kwargs
):
    # Positive callers only; original negative calls and raises remain verbatim.
    inputs = (ccol, row, values)
    try:
        candidate = flag_gems.sparse_csc_tensor
        assert callable(candidate)
        states = [_input_state(inp) for inp in inputs]
        assert all(inp.device == required_device for inp in inputs)
        assert kwargs["dtype"] == values.dtype and kwargs["layout"] == torch.sparse_csc
        assert kwargs["device"] == required_device
        parts = [(inp.dtype, tuple(inp.shape)) for inp in inputs]
        snapshots = [_storage_bytes(inp).cpu().clone() for inp in inputs]
    except Exception as error:
        raise AssertionError("Invalid CSC input ABI") from error
    try:
        result = candidate(ccol, row, values, *size, **kwargs)
    finally:
        try:
            for inp, state, before in zip(inputs, states, snapshots):
                assert _input_state(inp)[0] == state[0]
                assert torch.equal(_storage_bytes(inp).cpu(), before)
        except Exception as error:
            raise AssertionError("CSC constructor mutated an input") from error
    try:
        assert isinstance(result, torch.Tensor)
        assert result.device == required_device and result.dtype == kwargs["dtype"]
        assert result.layout == torch.sparse_csc and tuple(result.shape) == tuple(
            expected_shape
        )
        assert result.sparse_dim() == 2 and result.dense_dim() == 0
        for name, (dtype, shape) in zip(
            ("ccol_indices", "row_indices", "values"), parts
        ):
            _check_dense(getattr(result, name)(), dtype, shape, required_device)
    except Exception as error:
        raise AssertionError("Invalid CSC candidate output ABI") from error
    return result


def _assert_dense_equal(actual, expected, *, required_device, reference_device):
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    _check_dense(actual, expected.dtype, expected.shape, required_device)
    _check_dense(expected, actual.dtype, actual.shape, reference_device)
    # Preserve TU's exact tolerance and equal_nan policy after the original ABI.
    tu.assert_result_equal(_copy_tensor(actual, "cpu"), _copy_tensor(expected, "cpu"))


def _assert_result(
    res_out, ref_out, dtype, index_dtype, *, required_device, reference_device
):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.device == required_device and ref_out.device == reference_device
    assert res_out.layout == ref_out.layout == torch.sparse_csc
    assert res_out.dtype == ref_out.dtype == dtype
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.sparse_dim() == ref_out.sparse_dim() == 2
    assert res_out.dense_dim() == ref_out.dense_dim() == 0
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out)
    for result, reference in (
        (res_out.ccol_indices(), ref_out.ccol_indices()),
        (res_out.row_indices(), ref_out.row_indices()),
    ):
        _check_dense(result, index_dtype, reference.shape, required_device)
        _check_dense(reference, index_dtype, result.shape, reference_device)
        utils.gems_assert_equal(
            _copy_tensor(result, "cpu"), _copy_tensor(reference, "cpu")
        )
    _assert_dense_equal(
        res_out.values(),
        ref_out.values(),
        required_device=required_device,
        reference_device=reference_device,
    )


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _CSC_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8 or value_range in (["-1", "1"], ["0", "max"])
]


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize(
    "shape, nnz, dtype, value_range",
    [
        pytest.param(
            shape,
            nnz,
            dtype,
            value_range,
        )
        for shape, nnz in _CSC_CASES
        for dtype, value_range in _DTYPE_RANGE_CASES
    ],
)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test_sparse_csc_tensor(shape, nnz, dtype, index_dtype, value_range):
    required_device, reference_device = _required_devices()
    ccol, row, values = _make_csc_inputs(
        shape, nnz, dtype, index_dtype=index_dtype, value_range=value_range
    )

    ref_ccol = _to_reference(
        ccol, required_device=required_device, reference_device=reference_device
    )
    ref_row = _to_reference(
        row, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _call_candidate(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=required_device,
        required_device=required_device,
        expected_shape=tuple(ref_out.shape),
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        index_dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        ccol,
        ref_ccol,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        row, ref_row, required_device=required_device, reference_device=reference_device
    )
    _assert_dense_equal(
        values,
        ref_values,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape, nnz", _CSC_BATCHED_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize(
    "dtype, value_range",
    [
        pytest.param(
            dtype,
            value_range,
        )
        for dtype, value_range in _DTYPE_RANGE_CASES
    ],
)
def test_sparse_csc_tensor_batched(shape, nnz, dtype, index_dtype, value_range):
    required_device, reference_device = _required_devices()
    ccol, row, values = _make_csc_inputs(
        shape, nnz, dtype, index_dtype=index_dtype, value_range=value_range
    )

    ref_ccol = _to_reference(
        ccol, required_device=required_device, reference_device=reference_device
    )
    ref_row = _to_reference(
        row, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _call_candidate(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=required_device,
        required_device=required_device,
        expected_shape=tuple(ref_out.shape),
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        index_dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        ccol,
        ref_ccol,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        row, ref_row, required_device=required_device, reference_device=reference_device
    )
    _assert_dense_equal(
        values,
        ref_values,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize(
    "ccol_list, row_list, dtype, value_range",
    [
        pytest.param(
            ccol_list,
            row_list,
            dtype,
            value_range,
        )
        for ccol_list, row_list in _CSC_NO_SIZE_CASES
        for dtype, value_range in _DTYPE_RANGE_CASES
    ],
)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test_sparse_csc_tensor_no_size(
    ccol_list, row_list, dtype, index_dtype, value_range
):
    required_device, reference_device = _required_devices()
    ccol = torch.tensor(ccol_list, dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor(row_list, dtype=index_dtype, device=flag_gems.device)
    values = _make_values(len(row_list), dtype, value_range=value_range)

    ref_ccol = _to_reference(
        ccol, required_device=required_device, reference_device=reference_device
    )
    ref_row = _to_reference(
        row, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _call_candidate(
        ccol,
        row,
        values,
        dtype=dtype,
        layout=torch.sparse_csc,
        device=required_device,
        required_device=required_device,
        expected_shape=tuple(ref_out.shape),
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        index_dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        ccol,
        ref_ccol,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        row, ref_row, required_device=required_device, reference_device=reference_device
    )
    _assert_dense_equal(
        values,
        ref_values,
        required_device=required_device,
        reference_device=reference_device,
    )
    assert tuple(res_out.shape) == tuple(ref_out.shape)


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
        )
        for dtype in _CSC_DTYPES
    ],
)
def test_sparse_csc_tensor_unchecked_uncoalesced(dtype, index_dtype):
    required_device, reference_device = _required_devices()
    ccol = torch.tensor([0, 1, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([0, 0, 0], dtype=index_dtype, device=flag_gems.device)
    values = _make_values(3, dtype, value_range=["-1", "1"])

    ref_ccol = _to_reference(
        ccol, required_device=required_device, reference_device=reference_device
    )
    ref_row = _to_reference(
        row, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        [2, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _call_candidate(
        ccol,
        row,
        values,
        [2, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=required_device,
        required_device=required_device,
        expected_shape=tuple(ref_out.shape),
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        index_dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        ccol,
        ref_ccol,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        row, ref_row, required_device=required_device, reference_device=reference_device
    )
    _assert_dense_equal(
        values,
        ref_values,
        required_device=required_device,
        reference_device=reference_device,
    )
    assert torch.ops.aten._nnz(res_out) == 3


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
        )
        for dtype in _CSC_DTYPES
    ],
)
def test_sparse_csc_tensor_unchecked_unsorted_rows(dtype, index_dtype):
    required_device, reference_device = _required_devices()
    ccol = torch.tensor([0, 3, 3], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([1, 0, 2], dtype=index_dtype, device=flag_gems.device)
    values = _make_values(3, dtype, value_range=["-1", "1"])

    ref_ccol = _to_reference(
        ccol, required_device=required_device, reference_device=reference_device
    )
    ref_row = _to_reference(
        row, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        [3, 1],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _call_candidate(
        ccol,
        row,
        values,
        [3, 1],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=required_device,
        required_device=required_device,
        expected_shape=tuple(ref_out.shape),
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        index_dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        ccol,
        ref_ccol,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        row, ref_row, required_device=required_device, reference_device=reference_device
    )
    _assert_dense_equal(
        values,
        ref_values,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize("shape, nnz", _CSC_SHAPE_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize(
    "dtype, value_range",
    [
        pytest.param(
            dtype,
            value_range,
        )
        for dtype in _FLOAT_STORAGE_DTYPES
        for value_range in tu.selected_ranges()
    ],
)
def test_sparse_csc_tensor_shape_levels(shape, nnz, dtype, index_dtype, value_range):
    required_device, reference_device = _required_devices()
    ccol, row, values = _make_csc_inputs(
        shape, nnz, dtype, index_dtype=index_dtype, value_range=value_range
    )

    ref_ccol = _to_reference(
        ccol, required_device=required_device, reference_device=reference_device
    )
    ref_row = _to_reference(
        row, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _call_candidate(
        ccol,
        row,
        values,
        list(shape),
        dtype=dtype,
        layout=torch.sparse_csc,
        device=required_device,
        required_device=required_device,
        expected_shape=tuple(ref_out.shape),
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        index_dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        ccol,
        ref_ccol,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        row, ref_row, required_device=required_device, reference_device=reference_device
    )
    _assert_dense_equal(
        values,
        ref_values,
        required_device=required_device,
        reference_device=reference_device,
    )
    assert tuple(res_out.shape) == tuple(shape)


# Bool input generation ignores value_range; keep one sample per workload.
_BOUNDARY_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _CSC_DTYPES
    for value_range in (
        _BOUNDARY_RANGES[:1] if dtype == torch.bool else _BOUNDARY_RANGES
    )
]


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize(
    "dtype, value_range",
    [
        pytest.param(
            dtype,
            value_range,
        )
        for dtype, value_range in _BOUNDARY_DTYPE_RANGE_CASES
    ],
)
def test_sparse_csc_tensor_boundary_values(dtype, value_range):
    required_device, reference_device = _required_devices()
    ccol, row, values = _make_csc_inputs(
        (4, 4), 4, dtype, index_dtype=torch.int64, value_range=value_range
    )

    ref_ccol = _to_reference(
        ccol, required_device=required_device, reference_device=reference_device
    )
    ref_row = _to_reference(
        row, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        [4, 4],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _call_candidate(
        ccol,
        row,
        values,
        [4, 4],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=required_device,
        required_device=required_device,
        expected_shape=tuple(ref_out.shape),
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        torch.int64,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        ccol,
        ref_ccol,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        row, ref_row, required_device=required_device, reference_device=reference_device
    )
    _assert_dense_equal(
        values,
        ref_values,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csc_tensor
@pytest.mark.parametrize(
    "dtype, scenario",
    [
        pytest.param(
            dtype,
            scenario,
        )
        for dtype, scenario in tu.selected_cases(tu.special_value_cases(_CSC_DTYPES))
    ],
)
def test_sparse_csc_tensor_nan_inf_values(dtype, scenario):
    required_device, reference_device = _required_devices()
    values = _make_special_values(dtype, scenario, required_device=required_device)
    ccol = torch.tensor([0, 2, 5], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int64, device=flag_gems.device)

    ref_ccol = _to_reference(
        ccol, required_device=required_device, reference_device=reference_device
    )
    ref_row = _to_reference(
        row, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csc_tensor(
        ref_ccol,
        ref_row,
        ref_values,
        [3, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=ref_ccol.device,
    )
    res_out = _call_candidate(
        ccol,
        row,
        values,
        [3, 2],
        dtype=dtype,
        layout=torch.sparse_csc,
        device=required_device,
        required_device=required_device,
        expected_shape=tuple(ref_out.shape),
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        torch.int64,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        ccol,
        ref_ccol,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        row, ref_row, required_device=required_device, reference_device=reference_device
    )
    _assert_dense_equal(
        values,
        ref_values,
        required_device=required_device,
        reference_device=reference_device,
    )


def _build_default_inputs(dtype=torch.float32, index_dtype=torch.int64):
    values = _make_values(2, dtype, value_range=["0", "1"])
    ccol = torch.tensor([0, 1, 2], dtype=index_dtype, device=flag_gems.device)
    row = torch.tensor([0, 1], dtype=index_dtype, device=flag_gems.device)
    return ccol, row, values


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_dtype_mismatch():
    ccol, row, values = _build_default_inputs(dtype=torch.float16)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_missing_dtype():
    ccol, row, values = _build_default_inputs(dtype=torch.float64)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_wrong_layout():
    ccol, row, values = _build_default_inputs()
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
def test_sparse_csc_tensor_rejects_negative_size():
    ccol, row, values = _build_default_inputs()
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [-2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [-2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="cross-device construction requires a non-CPU device",
)
def test_sparse_csc_tensor_rejects_device_mismatch():
    ccol, row, values = _build_default_inputs()
    values = values.to("cpu")
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_csc_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="cross-device construction requires a non-CPU device",
)
def test_sparse_csc_tensor_rejects_missing_device():
    ccol, row, values = _build_default_inputs()
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_csc_tensor(
            ccol,
            row,
            values,
            [2, 2],
            dtype=torch.float32,
            layout=torch.sparse_csc,
        )
