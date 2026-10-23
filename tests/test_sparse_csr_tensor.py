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

# Build compressed sparse storage from index and value tensors.
# Pass dtype explicitly: these ATen factories default to float32.
# Accelerator inputs also need an explicit device.
_FP8_CSR_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_CSR_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + _FP8_CSR_DTYPES
    + [torch.int8, torch.uint8]
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)
_SHAPE_LEVEL_DTYPES = [
    torch.float32,
    torch.float8_e4m3fn,
    torch.int32,
]
_BOUNDARY_CSR_DTYPES = utils.ALL_FLOAT_DTYPES + _FP8_CSR_DTYPES

# (size, crow_indices, col_indices).
_CSR_2D_CASES = [
    ((4, 4), [0, 2, 4, 4, 4], [0, 1, 0, 1]),
    ((5, 4), [0, 2, 3, 3, 5, 5], [0, 1, 2, 0, 3]),
    ((3, 6), [0, 1, 3, 5], [0, 2, 4, 0, 5]),
    ((1, 4), [0, 2], [0, 3]),
    ((6, 1), [0, 1, 1, 2, 2, 3, 3], [0, 0, 0]),
    ((7, 5), [0, 1, 3, 4, 4, 6, 6, 6], [0, 2, 4, 1, 2, 3]),
    ((3, 3), [0, 3, 4, 6], [0, 1, 2, 0, 1, 2]),
]

# Batched matrices with a separate index grid per batch.
_CSR_3D_CASES = [
    ((2, 3, 4), [[0, 2, 3, 4], [0, 1, 3, 4]], [[0, 1, 0, 2], [1, 0, 2, 3]]),
    (
        (3, 4, 5),
        [[0, 1, 3, 4, 5], [0, 2, 2, 3, 5], [0, 1, 2, 4, 5]],
        [[0, 1, 3, 2, 4], [0, 4, 2, 1, 3], [1, 3, 0, 4, 2]],
    ),
    (
        (2, 5, 4),
        [[0, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 6]],
        [[0, 1, 3, 0, 2, 1], [3, 2, 0, 1, 0, 3]],
    ),
]

# (size, batch); None denotes an unbatched matrix.
_CSR_EMPTY_CASES = [
    ((4, 5), None),
    ((2, 4, 5), 2),
]

# (crow_indices, col_indices), with size inferred from indices.
_CSR_2D_INFERRED_CASES = [
    ([0, 2, 4], [0, 1, 0, 1]),
    ([0, 2, 2, 4], [0, 1, 0, 1]),
    ([0, 1, 3, 5], [0, 2, 4, 0, 5]),
    ([0, 1, 2], [0, 0]),
]

# Reject negative matrix dimensions.
_CSR_INVALID_SIZES = [[-1, 4], [4, -1], [-2, -2]]


def _range_for_dtype(dtype, value_range):
    # Clamp a negative lower bound to zero for unsigned storage.
    low_symbol, high_symbol = value_range
    if dtype != torch.bool and not dtype.is_floating_point:
        if torch.iinfo(dtype).min == 0 and low_symbol.startswith("-"):
            low_symbol = "0"
    return [low_symbol, high_symbol]


def _shape_level_cases():
    # Skip scalars, make 1-D shapes square, and generate seeded ragged rows.
    cases = []
    for shape in tu.selected_shapes():
        if len(shape) == 0:
            continue
        if len(shape) == 1:
            batch, rows, cols = (), shape[0], shape[0]
        else:
            batch, rows, cols = shape[:-2], shape[-2], shape[-1]
        gen = torch.Generator("cpu").manual_seed(len(shape))
        crow = [0]
        col = []
        for r in range(rows):
            k = min(1 + (r % 2), cols)
            chosen = torch.randperm(cols, generator=gen)[:k].sort().values
            col.extend(chosen.tolist())
            crow.append(crow[-1] + k)
        cases.append((batch + (rows, cols), crow, col))
    return cases


def _required_devices():
    target = torch.empty(0, dtype=torch.uint8, device=flag_gems.device).device
    return target, torch.device("cpu") if utils.TO_CPU else target


def _storage_bytes(inp):
    storage = inp.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _copy_tensor(inp, device):
    # Preserve physical Double/FP8 storage, offsets, padding and lazy flags.
    raw = _storage_bytes(inp).to(device=device).clone()
    out = raw.view(inp.dtype).as_strided(inp.shape, inp.stride(), inp.storage_offset())
    if inp.is_conj():
        out = out.conj()
    if inp.is_neg():
        out = torch._neg_view(out)
    out.requires_grad_(inp.requires_grad)
    return out


def _check_dense(inp, dtype, shape, device):
    assert isinstance(inp, torch.Tensor) and inp.layout == torch.strided
    assert inp.dtype == dtype and tuple(inp.shape) == tuple(shape)
    assert inp.device == device
    return inp


def _check_requested_indices(crow, col, *, dtype, size, nnz, required_device):
    # Derive requested geometry from the case, never from a potentially narrowed
    # or reshaped factory result shared by the candidate and reference.
    batch = tuple(size[:-2])
    _check_dense(crow, dtype, batch + (size[-2] + 1,), required_device)
    _check_dense(col, dtype, batch + (nnz,), required_device)


def _to_reference(inp, *, required_device, reference_device):
    assert isinstance(inp, torch.Tensor) and inp.device == required_device
    out = _copy_tensor(inp, reference_device)
    _check_dense(out, inp.dtype, inp.shape, reference_device)
    assert out.stride() == inp.stride() and out.storage_offset() == inp.storage_offset()
    assert out.is_conj() == inp.is_conj() and out.is_neg() == inp.is_neg()
    assert out.untyped_storage()._cdata != inp.untyped_storage()._cdata
    return out


def _input_state(inp):
    assert isinstance(inp, torch.Tensor) and inp.layout == torch.strided
    storage = inp.untyped_storage()
    return (
        (
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
        ),
        storage,
        inp._base,
    )


def _call_candidate(crow, col, values, *size, required_device, **kwargs):
    # Guard faults must not satisfy a negative caller's expected native error.
    try:
        inputs = (crow, col, values)
        states = [_input_state(inp) for inp in inputs]
        assert crow.device == col.device == required_device
        snapshots = [_storage_bytes(inp).cpu().clone() for inp in inputs]
        parts = [(inp.dtype, tuple(inp.shape)) for inp in inputs]
        candidate = flag_gems.sparse_csr_tensor
        assert callable(candidate)
    except Exception as error:
        raise AssertionError("Invalid CSR constructor input ABI") from error
    try:
        result = candidate(crow, col, values, *size, **kwargs)
    finally:
        try:
            for inp, state, before in zip(inputs, states, snapshots):
                assert _input_state(inp)[0] == state[0]
                assert torch.equal(_storage_bytes(inp).cpu(), before)
        except Exception as error:
            raise AssertionError("CSR constructor mutated an input") from error
    try:
        assert isinstance(result, torch.Tensor) and result.layout == torch.sparse_csr
        assert result.device == required_device
        if "dtype" in kwargs:
            assert result.dtype == kwargs["dtype"]
        for part, (dtype, shape) in zip(
            (result.crow_indices(), result.col_indices(), result.values()), parts
        ):
            _check_dense(part, dtype, shape, required_device)
    except Exception as error:
        raise AssertionError("Invalid CSR candidate output ABI") from error
    return result


def _assert_dense_equal(
    actual, expected, *, required_device, reference_device, equal_nan=False
):
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    assert actual.device == required_device and expected.device == reference_device
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.layout == expected.layout == torch.strided
    actual, expected = _copy_tensor(actual, "cpu"), _copy_tensor(expected, "cpu")
    if actual.dtype in _FP8_CSR_DTYPES:
        actual, expected = actual.float(), expected.float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=equal_nan)


def _make_special_values(dtype, scenario, nnz, *, required_device):
    if dtype in _FP8_CSR_DTYPES or (
        flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype == torch.float64
    ):
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        host = torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu").to(
            dtype
        )
        values = _copy_tensor(host[:nnz], required_device)
    else:
        values = tu.make_special_input(dtype, scenario)[:nnz]
    return _check_dense(values, dtype, (nnz,), required_device)


def _make_boundary_values(dtype, nnz, *, required_device):
    finfo = torch.finfo(dtype)
    if dtype in _FP8_CSR_DTYPES or (
        flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype == torch.float64
    ):
        specials = torch.tensor(
            [finfo.min, finfo.max, 0.0, -0.0, 1.0, -1.0], dtype=dtype, device="cpu"
        )
        # Byte repetition also works on older CPU FP8 implementations.
        raw = specials.view(torch.uint8).repeat((nnz + 5) // 6)
        values = _copy_tensor(raw.view(dtype)[:nnz], required_device)
    else:
        specials = torch.tensor(
            [finfo.min, finfo.max, 0.0, -0.0, 1.0, -1.0],
            dtype=dtype,
            device=flag_gems.device,
        )
        values = specials.repeat((nnz + specials.numel() - 1) // specials.numel())[:nnz]
    return _check_dense(values, dtype, (nnz,), required_device)


def _make_csr_values(
    nnz, dtype, batch=None, value_range=("-1", "1"), *, required_device
):
    if batch is None:
        shape = (nnz,)
    elif isinstance(batch, tuple):
        shape = batch + (nnz,)
    else:
        shape = (batch,) + (nnz,)
    bounds = _range_for_dtype(dtype, value_range)
    if dtype in _FP8_CSR_DTYPES or (
        flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype == torch.float64
    ):
        low, high = (tu.resolve_bound(bound, dtype) for bound in bounds)
        if low == high:
            host = torch.full(shape, low, dtype=dtype, device="cpu")
        else:
            host = torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        values = _copy_tensor(host, required_device)
    else:
        values = tu.make_input(dtype, shape, bounds)
    return _check_dense(values, dtype, shape, required_device)


def _assert_csr_structure(out, size, nnz, dtype, batch=None, *, required_device):
    assert isinstance(out, torch.Tensor)
    assert out.device == required_device
    assert out.layout == torch.sparse_csr
    assert tuple(out.shape) == tuple(size)
    assert out.dtype == dtype
    assert out.sparse_dim() == 2
    assert out.dense_dim() == 0
    assert out._nnz() == nnz
    expected_crow_len = size[-2] + 1
    if batch is None:
        assert tuple(out.values().shape) == (nnz,)
        assert len(out.crow_indices()) == expected_crow_len
        assert len(out.col_indices()) == nnz
    elif isinstance(batch, tuple):
        assert tuple(out.values().shape) == batch + (nnz,)
        assert tuple(out.crow_indices().shape) == batch + (expected_crow_len,)
        assert tuple(out.col_indices().shape) == batch + (nnz,)
    else:
        assert tuple(out.values().shape) == (batch, nnz)
        assert tuple(out.crow_indices().shape) == (batch, expected_crow_len)
        assert tuple(out.col_indices().shape) == (batch, nnz)
    for part in (out.crow_indices(), out.col_indices(), out.values()):
        assert isinstance(part, torch.Tensor) and part.device == required_device
    assert out.values().dtype == dtype
    cols = _copy_tensor(out.col_indices(), "cpu")
    assert (cols >= 0).all()
    assert (cols < size[-1]).all()


def _assert_csr_equal(res_out, ref_out, *, required_device, reference_device):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.device == required_device and ref_out.device == reference_device
    assert res_out.layout == ref_out.layout == torch.sparse_csr
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.dtype == ref_out.dtype
    _assert_dense_equal(
        res_out.crow_indices(),
        ref_out.crow_indices(),
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        res_out.col_indices(),
        ref_out.col_indices(),
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        res_out.values(),
        ref_out.values(),
        required_device=required_device,
        reference_device=reference_device,
        equal_nan=True,
    )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_CASES)
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csr_tensor_crow_col_value_size(case, dtype, value_range):
    required_device, reference_device = _required_devices()
    size, crow, col = case
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(
        nnz, dtype, value_range=value_range, required_device=required_device
    )
    _check_requested_indices(
        crow_t,
        col_t,
        dtype=torch.long,
        size=size,
        nnz=nnz,
        required_device=required_device,
    )
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=crow_t.device,
        required_device=required_device,
    )
    _assert_csr_structure(res_out, size, nnz, dtype, required_device=required_device)
    _assert_csr_equal(
        res_out,
        ref_out,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        crow_t,
        ref_crow,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        col_t,
        ref_col,
        required_device=required_device,
        reference_device=reference_device,
    )
    _assert_dense_equal(
        values,
        ref_values,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_3D_CASES)
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csr_tensor_crow_col_value_size_batched(case, dtype, value_range):
    required_device, reference_device = _required_devices()
    size, crow, col = case
    batch = size[0]
    nnz = len(col[0])
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(
        nnz,
        dtype,
        batch=batch,
        value_range=value_range,
        required_device=required_device,
    )
    _check_requested_indices(
        crow_t,
        col_t,
        dtype=torch.long,
        size=size,
        nnz=nnz,
        required_device=required_device,
    )
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=crow_t.device,
        required_device=required_device,
    )
    _assert_csr_structure(
        res_out, size, nnz, dtype, batch=batch, required_device=required_device
    )
    _assert_csr_equal(
        res_out,
        ref_out,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_empty(case, dtype):
    required_device, reference_device = _required_devices()
    size, batch = case
    n_rows = size[-2]
    if batch is None:
        crow_t = torch.zeros(n_rows + 1, dtype=torch.long, device=flag_gems.device)
        col_t = torch.empty(0, dtype=torch.long, device=flag_gems.device)
        values = _make_csr_values(0, dtype, required_device=required_device)
    else:
        crow_t = torch.zeros(
            batch, n_rows + 1, dtype=torch.long, device=flag_gems.device
        )
        col_t = torch.empty(batch, 0, dtype=torch.long, device=flag_gems.device)
        values = _make_csr_values(
            0, dtype, batch=batch, required_device=required_device
        )
    _check_requested_indices(
        crow_t,
        col_t,
        dtype=torch.long,
        size=size,
        nnz=0,
        required_device=required_device,
    )
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=crow_t.device,
        required_device=required_device,
    )
    _assert_csr_structure(
        res_out, size, 0, dtype, batch=batch, required_device=required_device
    )
    _assert_csr_equal(
        res_out,
        ref_out,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_CASES[:3])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_index_dtypes(case, index_dtype, dtype):
    required_device, reference_device = _required_devices()
    size, crow, col = case
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=index_dtype, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=index_dtype, device=flag_gems.device)
    values = _make_csr_values(nnz, dtype, required_device=required_device)
    _check_requested_indices(
        crow_t,
        col_t,
        dtype=index_dtype,
        size=size,
        nnz=nnz,
        required_device=required_device,
    )
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=crow_t.device,
        required_device=required_device,
    )
    _assert_csr_structure(res_out, size, nnz, dtype, required_device=required_device)
    assert res_out.crow_indices().dtype == index_dtype
    assert res_out.col_indices().dtype == index_dtype
    _assert_csr_equal(
        res_out,
        ref_out,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
def test_sparse_csr_tensor_crow_col_value_size_trailing_empty_rows(dtype):
    required_device, reference_device = _required_devices()
    size = (5, 2)
    crow = [0, 2, 4, 4, 4, 4]
    col = [0, 1, 0, 1]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, dtype, required_device=required_device)
    _check_requested_indices(
        crow_t,
        col_t,
        dtype=torch.long,
        size=size,
        nnz=nnz,
        required_device=required_device,
    )
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=crow_t.device,
        required_device=required_device,
    )
    _assert_csr_structure(res_out, size, nnz, dtype, required_device=required_device)
    _assert_csr_equal(
        res_out,
        ref_out,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _CSR_2D_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _CSR_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csr_tensor_crow_col_value(case, dtype, value_range):
    required_device, reference_device = _required_devices()
    crow, col = case
    nnz = len(col)
    size = (len(crow) - 1, max(col) + 1)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(
        nnz, dtype, value_range=value_range, required_device=required_device
    )
    _check_requested_indices(
        crow_t,
        col_t,
        dtype=torch.long,
        size=size,
        nnz=nnz,
        required_device=required_device,
    )
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        dtype=dtype,
        device=crow_t.device,
        required_device=required_device,
    )
    _assert_csr_structure(res_out, size, nnz, dtype, required_device=required_device)
    _assert_csr_equal(
        res_out,
        ref_out,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("case", _shape_level_cases())
@pytest.mark.parametrize("dtype", _SHAPE_LEVEL_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_csr_tensor_shape_levels(case, dtype, value_range):
    required_device, reference_device = _required_devices()
    size, crow, col = case
    batch = size[:-2] or None
    nnz = len(col)
    if batch is None:
        crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
        col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    else:
        crow_t = (
            torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
            .expand(batch + (len(crow),))
            .contiguous()
        )
        col_t = (
            torch.tensor(col, dtype=torch.long, device=flag_gems.device)
            .expand(batch + (nnz,))
            .contiguous()
        )
    values = _make_csr_values(
        nnz,
        dtype,
        batch=batch,
        value_range=value_range,
        required_device=required_device,
    )
    _check_requested_indices(
        crow_t,
        col_t,
        dtype=torch.long,
        size=size,
        nnz=nnz,
        required_device=required_device,
    )
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=crow_t.device,
        required_device=required_device,
    )
    _assert_csr_structure(
        res_out, size, nnz, dtype, batch=batch, required_device=required_device
    )
    _assert_csr_equal(
        res_out,
        ref_out,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_CSR_DTYPES))
)
def test_sparse_csr_tensor_nan_inf_values(dtype, scenario):
    required_device, reference_device = _required_devices()
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    values = _make_special_values(dtype, scenario, nnz, required_device=required_device)
    crow_t = torch.tensor([0, 2, 4, 4, 4], dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=flag_gems.device)
    _check_requested_indices(
        crow_t,
        col_t,
        dtype=torch.long,
        size=size,
        nnz=nnz,
        required_device=required_device,
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_values.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=values.device,
        required_device=required_device,
    )
    _assert_csr_structure(res_out, size, nnz, dtype, required_device=required_device)
    _assert_csr_equal(
        res_out,
        ref_out,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("dtype", _BOUNDARY_CSR_DTYPES)
def test_sparse_csr_tensor_boundary_values(dtype):
    required_device, reference_device = _required_devices()
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    values = _make_boundary_values(dtype, nnz, required_device=required_device)
    crow_t = torch.tensor([0, 2, 4, 4, 4], dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=flag_gems.device)
    _check_requested_indices(
        crow_t,
        col_t,
        dtype=torch.long,
        size=size,
        nnz=nnz,
        required_device=required_device,
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_out = torch.ops.aten.sparse_csr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_values.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=values.device,
        required_device=required_device,
    )
    _assert_csr_structure(res_out, size, nnz, dtype, required_device=required_device)
    _assert_csr_equal(
        res_out,
        ref_out,
        required_device=required_device,
        reference_device=reference_device,
    )


@pytest.mark.sparse_csr_tensor
def test_sparse_csr_tensor_rejects_dtype_mismatch():
    required_device, reference_device = _required_devices()
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, torch.float16, required_device=required_device)
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csr_tensor(
            ref_crow,
            ref_col,
            ref_values,
            list(size),
            dtype=torch.float32,
            device=ref_crow.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _call_candidate(
            crow_t,
            col_t,
            values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
            required_device=required_device,
        )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.int32]
)
def test_sparse_csr_tensor_rejects_missing_dtype(dtype):
    required_device, reference_device = _required_devices()
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, dtype, required_device=required_device)
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csr_tensor(
            ref_crow, ref_col, ref_values, list(size), device=ref_crow.device
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _call_candidate(
            crow_t,
            col_t,
            values,
            list(size),
            device=crow_t.device,
            required_device=required_device,
        )


@pytest.mark.sparse_csr_tensor
@pytest.mark.parametrize("invalid_size", _CSR_INVALID_SIZES)
def test_sparse_csr_tensor_rejects_invalid_size(invalid_size):
    required_device, reference_device = _required_devices()
    crow = [0, 2, 4, 4, 4]
    col = [0, 1, 0, 1]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, torch.float32, required_device=required_device)
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csr_tensor(
            ref_crow,
            ref_col,
            ref_values,
            list(invalid_size),
            dtype=torch.float32,
            device=ref_crow.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _call_candidate(
            crow_t,
            col_t,
            values,
            list(invalid_size),
            dtype=torch.float32,
            device=crow_t.device,
            required_device=required_device,
        )


@pytest.mark.sparse_csr_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="cross-device construction requires a non-CPU device",
)
def test_sparse_csr_tensor_rejects_device_mismatch():
    required_device, reference_device = _required_devices()
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, torch.float32, required_device=required_device)
    cpu_values = values.cpu()
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csr_tensor(
            crow_t,
            col_t,
            cpu_values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _call_candidate(
            crow_t,
            col_t,
            cpu_values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
            required_device=required_device,
        )


@pytest.mark.sparse_csr_tensor
@pytest.mark.skipif(
    flag_gems.device != "cuda", reason="the missing-device quirk only exists on CUDA"
)
def test_sparse_csr_tensor_rejects_missing_device():
    required_device, reference_device = _required_devices()
    size, crow, col = _CSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_csr_values(nnz, torch.float32, required_device=required_device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_csr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        _call_candidate(
            crow_t,
            col_t,
            values,
            list(size),
            dtype=torch.float32,
            required_device=required_device,
        )
