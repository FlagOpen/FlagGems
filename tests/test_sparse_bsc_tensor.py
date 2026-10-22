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

# Select every overload and contract case with the operator marker.
pytestmark = pytest.mark.sparse_bsc_tensor

# Build compressed sparse storage from index and value tensors.
# Pass dtype explicitly: these ATen factories default to float32.
# Accelerator inputs also need an explicit device.
_BSC_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)
_INDEX_DTYPES = [torch.int32, torch.int64]

# (matrix_shape, block_shape, nnz).
_BSC_CASES = [
    ((4, 4), (2, 2), 3),  # 2x2 row/col blocks, partial fill
    ((8, 8), (2, 2), 16),  # full 4x4 block grid
    ((16, 16), (4, 4), 8),  # larger blocks
    ((6, 8), (2, 2), 6),  # non-square matrix
    ((6, 6), (3, 3), 4),  # 3x3 blocks
    ((2, 6), (2, 3), 2),  # single row block
    ((4, 4), (2, 2), 0),  # empty (nnz == 0)
]

_BSC_VALUE_CASES = [
    ((4, 4), (2, 2), 3),
    ((6, 8), (2, 2), 6),
]

# Legacy 1-D values cannot be densified; compare the stored structure.
_LEGACY_CASES = [
    ((4, 5), 3),
    ((4, 5), 0),
]


def _bsc_shape_level_cases():
    # Use trailing matrix dimensions and blocks that divide each extent.
    cases = []
    for shape in tu.selected_shapes():
        shape = tuple(shape)
        if len(shape) < 2:
            continue
        nrows, ncols = shape[-2], shape[-1]
        block = (2 if nrows % 2 == 0 else 1, 2 if ncols % 2 == 0 else 1)
        n_blocks = (nrows // block[0]) * (ncols // block[1])
        nnz = min(6, max(1, n_blocks))
        cases.append(((nrows, ncols), block, nnz))
    if not cases:
        cases = [((4, 4), (2, 2), 4)]
    return cases


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _required_devices():
    # Resolve the logical target before a candidate can change ambient state.
    target = torch.empty(0, dtype=torch.uint8, device=flag_gems.device).device
    return target, torch.device("cpu") if utils.TO_CPU else target


def _storage_bytes(inp):
    storage = inp.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _copy_tensor(inp, device):
    # Typed .to() can narrow Double on KU; transfer bytes before restoring dtype.
    raw = _storage_bytes(inp).to(device=device).clone()
    out = raw.view(inp.dtype).as_strided(inp.shape, inp.stride(), inp.storage_offset())
    if inp.is_conj():
        out = out.conj()
    if inp.is_neg():
        out = torch._neg_view(out)
    out.requires_grad_(inp.requires_grad)
    return out


def _check_dense(inp, dtype, shape, device):
    assert isinstance(inp, torch.Tensor)
    assert inp.layout == torch.strided
    assert inp.dtype == dtype
    assert tuple(inp.shape) == tuple(shape)
    assert inp.device == device
    return inp


def _make_index(data, dtype, *, required_device):
    inp = torch.tensor(data, dtype=dtype, device=required_device)
    return _check_dense(inp, dtype, (len(data),), required_device)


def _cpu_encode(dtype):
    return (
        flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES
    ) or (flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64)


def _make_values(dtype, shape, value_range, *, required_device):
    if not _cpu_encode(dtype):
        values = tu.make_input(dtype, shape, value_range).to(required_device)
    else:
        # Preserve TU's original range and make_tensor recipe on CPU.
        low = tu.resolve_bound(value_range[0], dtype)
        high = tu.resolve_bound(value_range[1], dtype)
        if low == high:
            host = torch.full(shape, low, dtype=dtype, device="cpu")
        else:
            host = torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        values = _copy_tensor(host, required_device)
    return _check_dense(values, dtype, shape, required_device)


def _special_values(dtype, scenario, *, required_device):
    if not _cpu_encode(dtype):
        values = tu.make_special_input(dtype, scenario).repeat(4)[:18].reshape(2, 3, 3)
    else:
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        host = torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu")
        host = host.to(dtype).repeat(4)[:18].reshape(2, 3, 3)
        values = _copy_tensor(host, required_device)
    return _check_dense(values, dtype, (2, 3, 3), required_device)


def _to_reference(inp, *, required_device, reference_device):
    try:
        assert isinstance(inp, torch.Tensor) and inp.device == required_device
        out = _copy_tensor(inp, reference_device)
        _check_dense(out, inp.dtype, inp.shape, reference_device)
        assert out.stride() == inp.stride()
        assert out.storage_offset() == inp.storage_offset()
        assert out.is_conj() == inp.is_conj() and out.is_neg() == inp.is_neg()
        assert out.untyped_storage()._cdata != inp.untyped_storage()._cdata
    except Exception as error:
        raise AssertionError("Invalid BSC reference transport") from error
    return out


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


def _make_bsc_structure(shape, block, nnz, index_dtype=torch.int64, *, required_device):
    M, N = shape
    Br, Bc = block
    n_row_blocks, n_col_blocks = M // Br, N // Bc
    assert M % Br == N % Bc == 0
    assert 0 <= nnz <= n_row_blocks * n_col_blocks
    counts = torch.full((n_col_blocks,), nnz // n_col_blocks, dtype=torch.long)
    counts[: nnz % n_col_blocks] += 1
    ccol = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    row = torch.arange(nnz) - torch.repeat_interleave(ccol[:-1], counts)
    ccol = ccol.to(device=required_device, dtype=index_dtype)
    row = row.to(device=required_device, dtype=index_dtype)
    _check_dense(ccol, index_dtype, (n_col_blocks + 1,), required_device)
    _check_dense(row, index_dtype, (nnz,), required_device)
    return ccol, row


def _make_bsc_inputs(
    shape, block, nnz, dtype, value_range, index_dtype=torch.int64, *, required_device
):
    ccol, row = _make_bsc_structure(
        shape, block, nnz, index_dtype=index_dtype, required_device=required_device
    )
    values = _make_values(
        dtype, (nnz,) + tuple(block), value_range, required_device=required_device
    )
    return ccol, row, values


def _call_reference(
    ccol, row, values, size, dtype, *, required_device, reference_device
):
    ref_ccol, ref_row, ref_values = (
        _to_reference(
            inp, required_device=required_device, reference_device=reference_device
        )
        for inp in (ccol, row, values)
    )
    result = torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
        ref_ccol,
        ref_row,
        ref_values,
        size=list(size),
        dtype=dtype,
        layout=torch.sparse_bsc,
        device=ref_ccol.device,
    )
    assert isinstance(result, torch.Tensor)
    assert result.device == reference_device and result.dtype == dtype
    assert result.layout == torch.sparse_bsc and tuple(result.shape) == tuple(size)
    return result


def _call_candidate(ccol, row, values, size, dtype, *, required_device):
    # Inspection errors must not satisfy a negative caller's native raises block.
    inputs = (ccol, row, values)
    try:
        states = [_input_state(inp) for inp in inputs]
        assert all(inp.device == required_device for inp in inputs)
        snapshots = [_storage_bytes(inp).cpu().clone() for inp in inputs]
    except Exception as error:
        raise AssertionError("Invalid BSC input state before candidate") from error
    try:
        result = flag_gems.sparse_bsc_tensor(
            ccol,
            row,
            values,
            size=list(size),
            dtype=dtype,
            layout=torch.sparse_bsc,
            device=required_device,
        )
    finally:
        try:
            for inp, state, before in zip(inputs, states, snapshots):
                assert _input_state(inp)[0] == state[0]
                assert torch.equal(_storage_bytes(inp).cpu(), before)
        except Exception as error:
            raise AssertionError("BSC constructor mutated an input") from error
    try:
        assert isinstance(result, torch.Tensor)
        assert result.device == required_device and result.dtype == dtype
        assert result.layout == torch.sparse_bsc and tuple(result.shape) == tuple(size)
    except Exception as error:
        raise AssertionError("Invalid BSC candidate output ABI") from error
    return result


def _assert_result(res_out, ref_out, dtype, *, required_device, reference_device):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.device == required_device and ref_out.device == reference_device
    assert res_out.layout == ref_out.layout == torch.sparse_bsc
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.sparse_dim() == ref_out.sparse_dim()
    assert res_out.dense_dim() == ref_out.dense_dim()
    assert res_out.shape == ref_out.shape
    for name in ("ccol_indices", "row_indices", "values"):
        actual, expected = getattr(res_out, name)(), getattr(ref_out, name)()
        _check_dense(actual, expected.dtype, expected.shape, required_device)
        _check_dense(expected, actual.dtype, actual.shape, reference_device)
        assert actual.is_conj() == expected.is_conj()
        assert actual.is_neg() == expected.is_neg()
        actual, expected = _copy_tensor(actual, "cpu"), _copy_tensor(expected, "cpu")
        if name == "values":
            tu.assert_result_equal(actual, expected)
        else:
            utils.gems_assert_equal(actual, expected)


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _BSC_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor(case, dtype, index_dtype):
    required_device, reference_device = _required_devices()
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(
        shape,
        block,
        nnz,
        dtype,
        ["-1", "1"],
        index_dtype=index_dtype,
        required_device=required_device,
    )

    ref_out = _call_reference(
        ccol,
        row,
        values,
        shape,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        ccol, row, values, shape, dtype, required_device=required_device
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _bsc_shape_level_cases())
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_shape_levels(case, dtype, index_dtype):
    required_device, reference_device = _required_devices()
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(
        shape,
        block,
        nnz,
        dtype,
        ["-1", "1"],
        index_dtype=index_dtype,
        required_device=required_device,
    )

    ref_out = _call_reference(
        ccol,
        row,
        values,
        shape,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        ccol, row, values, shape, dtype, required_device=required_device
    )

    assert tuple(res_out.shape) == tuple(shape)
    _assert_result(
        res_out,
        ref_out,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _BSC_VALUE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_value_ranges(case, value_range, dtype):
    required_device, reference_device = _required_devices()
    shape, block, nnz = case
    ccol, row, values = _make_bsc_inputs(
        shape, block, nnz, dtype, value_range, required_device=required_device
    )

    ref_out = _call_reference(
        ccol,
        row,
        values,
        shape,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        ccol, row, values, shape, dtype, required_device=required_device
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_BSC_DTYPES))
)
def test_sparse_bsc_tensor_nan_inf(dtype, scenario):
    required_device, reference_device = _required_devices()
    values = _special_values(dtype, scenario, required_device=required_device)
    ccol = _make_index([0, 1, 2], torch.int64, required_device=required_device)
    row = _make_index([0, 1], torch.int64, required_device=required_device)

    ref_out = _call_reference(
        ccol,
        row,
        values,
        [6, 6],
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        ccol, row, values, [6, 6], dtype, required_device=required_device
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("case", _LEGACY_CASES)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_unchecked_legacy(case, dtype, index_dtype):
    required_device, reference_device = _required_devices()
    shape, nnz = case
    ccol, row = _make_bsc_structure(
        shape, (1, 1), nnz, index_dtype=index_dtype, required_device=required_device
    )
    values = _make_values(dtype, (nnz,), ["-1", "1"], required_device=required_device)

    ref_out = _call_reference(
        ccol,
        row,
        values,
        shape,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        ccol, row, values, shape, dtype, required_device=required_device
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_unchecked_uncoalesced(dtype, index_dtype):
    required_device, reference_device = _required_devices()
    ccol = _make_index([0, 2, 3], index_dtype, required_device=required_device)
    row = _make_index([0, 0, 1], index_dtype, required_device=required_device)
    values = _make_values(
        dtype, (3, 2, 2), ["-1", "1"], required_device=required_device
    )

    ref_out = _call_reference(
        ccol,
        row,
        values,
        [4, 4],
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        ccol, row, values, [4, 4], dtype, required_device=required_device
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_bsc_tensor
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", _BSC_DTYPES)
def test_sparse_bsc_tensor_unchecked_unsorted_rows(dtype, index_dtype):
    required_device, reference_device = _required_devices()
    ccol = _make_index([0, 3, 3], index_dtype, required_device=required_device)
    row = _make_index([1, 0, 1], index_dtype, required_device=required_device)
    values = _make_values(
        dtype, (3, 2, 2), ["-1", "1"], required_device=required_device
    )

    ref_out = _call_reference(
        ccol,
        row,
        values,
        [4, 4],
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )
    res_out = _call_candidate(
        ccol, row, values, [4, 4], dtype, required_device=required_device
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_dtype_mismatch():
    required_device, reference_device = _required_devices()
    ccol = _make_index([0, 2, 3], torch.int64, required_device=required_device)
    row = _make_index([0, 1, 1], torch.int64, required_device=required_device)
    values = _make_values(
        torch.float64, (3, 2, 2), ["-1", "1"], required_device=required_device
    )

    with pytest.raises(RuntimeError):
        _call_reference(
            ccol,
            row,
            values,
            [4, 4],
            torch.float32,
            required_device=required_device,
            reference_device=reference_device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _call_candidate(
            ccol, row, values, [4, 4], torch.float32, required_device=required_device
        )


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_layout():
    required_device, reference_device = _required_devices()
    ccol = _make_index([0, 2, 3], torch.int64, required_device=required_device)
    row = _make_index([0, 1, 1], torch.int64, required_device=required_device)
    values = _make_values(
        torch.float32, (3, 2, 2), ["-1", "1"], required_device=required_device
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

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
            ref_ccol,
            ref_row,
            ref_values,
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=ref_ccol.device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_bsc_tensor(
            ccol,
            row,
            values,
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=flag_gems.device,
        )


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_size():
    required_device, reference_device = _required_devices()
    ccol = _make_index([0, 2, 3], torch.int64, required_device=required_device)
    row = _make_index([0, 1, 1], torch.int64, required_device=required_device)
    values = _make_values(
        torch.float32, (3, 2, 2), ["-1", "1"], required_device=required_device
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

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
            ref_ccol,
            ref_row,
            ref_values,
            size=[-4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=ref_ccol.device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_bsc_tensor(
            ccol,
            row,
            values,
            size=[-4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=flag_gems.device,
        )


@pytest.mark.sparse_bsc_tensor_negative
def test_sparse_bsc_tensor_negative_non_tensor():
    required_device, reference_device = _required_devices()
    row = _make_index([0, 1, 1], torch.int64, required_device=required_device)
    values = _make_values(
        torch.float32, (3, 2, 2), ["-1", "1"], required_device=required_device
    )

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsc_tensor.ccol_row_value_size(
            3.14,
            _to_reference(
                row, required_device=required_device, reference_device=reference_device
            ),
            _to_reference(
                values,
                required_device=required_device,
                reference_device=reference_device,
            ),
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_bsc_tensor(
            3.14,
            row,
            values,
            size=[4, 4],
            dtype=torch.float32,
            layout=torch.sparse_bsc,
            device=flag_gems.device,
        )
