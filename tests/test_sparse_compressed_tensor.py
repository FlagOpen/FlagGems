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


# Build compressed sparse storage from index and value tensors.
# Pass dtype explicitly: these ATen factories default to float32.
# Accelerator inputs also need an explicit device.
_SPARSE_COMPRESSED_DTYPES = (
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)
_BLOCK_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)
_BLOCK_SIZE = 2

# (layout, shape, nnz, index_dtype).
_LAYOUT_CASES = tu.selected_cases(
    [
        (torch.sparse_csr, (5, 4), 7, torch.int64),
        (torch.sparse_csc, (5, 4), 7, torch.int64),
        (torch.sparse_bsr, (6, 4), 6, torch.int64),
        (torch.sparse_bsc, (4, 6), 6, torch.int64),
        (torch.sparse_csr, (2, 3, 5, 4), 8, torch.int64),
        (torch.sparse_csr, (3, 5, 4), 7, torch.int32),
        (torch.sparse_bsr, (2, 4, 6, 4), 6, torch.int64),
        (torch.sparse_csc, (3, 4, 5, 6), 9, torch.int32),
    ],
    quick=[(torch.sparse_csr, (2, 19, 7), 12, torch.int64)],
)

# (layout, shape, nnz, index_dtype).
_VALUE_CASES = tu.selected_cases(
    [
        (torch.sparse_csr, (5, 4), 7, torch.int64),
        (torch.sparse_bsr, (6, 4), 6, torch.int64),
        (torch.sparse_csr, (2, 3, 5, 4), 8, torch.int64),
        (torch.sparse_bsc, (2, 4, 4, 6), 6, torch.int64),
    ],
    quick=[(torch.sparse_csr, (2, 19, 7), 12, torch.int64)],
)

# Layouts exercised by the size-inferred overload.
_NO_SIZE_CASES = [
    (torch.sparse_csr, (5, 4), 7, torch.int64),
    (torch.sparse_csc, (5, 4), 7, torch.int64),
    (torch.sparse_bsr, (6, 4), 6, torch.int64),
]

# Empty storage for each compressed layout.
_EMPTY_CASES = [
    (torch.sparse_csr, (4, 5)),
    (torch.sparse_csc, (4, 5)),
    (torch.sparse_bsr, (4, 4)),
    (torch.sparse_bsc, (4, 4)),
]


def _range_for_dtype(dtype, value_range):
    # Clamp a negative lower bound to zero for unsigned storage.
    low_symbol, high_symbol = value_range
    if dtype != torch.bool and not dtype.is_floating_point:
        if torch.iinfo(dtype).min == 0 and low_symbol.startswith("-"):
            low_symbol = "0"
    return [low_symbol, high_symbol]


def _required_devices():
    target = torch.empty(0, dtype=torch.uint8, device=flag_gems.device).device
    return target, torch.device("cpu") if utils.TO_CPU else target


def _storage_bytes(inp):
    storage = inp.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _copy_tensor(inp, device):
    # Physical bytes avoid unsupported FP8 kernels and typed Double narrowing.
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


def _make_values(dtype, shape, value_range, *, required_device):
    if dtype not in _FP8_DTYPES:
        values = tu.make_input(dtype, shape, value_range).to(required_device)
    else:
        # Same TU bounds and make_tensor/full recipe, encoded on CPU first.
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


def _make_special_values(dtype, scenario, *, required_device):
    if dtype in _FP8_DTYPES or (
        flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype == torch.float64
    ):
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        host = torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu")
        values = _copy_tensor(host.to(dtype).repeat(2)[:7], required_device)
    else:
        values = tu.make_special_input(dtype, scenario).repeat(2)[:7]
    return _check_dense(values, dtype, (7,), required_device)


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
    compressed, plain, values, *size, required_device, expected_shape, **kwargs
):
    # Positive callers only. Original negative native calls keep their boundaries.
    inputs = (compressed, plain, values)
    try:
        states = [_input_state(inp) for inp in inputs]
        assert all(inp.device == required_device for inp in inputs)
        dtype, layout = kwargs["dtype"], kwargs["layout"]
        assert kwargs["device"] == required_device and values.dtype == dtype
        snapshots = [_storage_bytes(inp).cpu().clone() for inp in inputs]
        parts = [(inp.dtype, tuple(inp.shape)) for inp in inputs]
    except Exception as error:
        raise AssertionError("Invalid compressed constructor input ABI") from error
    try:
        result = flag_gems.sparse_compressed_tensor(
            compressed, plain, values, *size, **kwargs
        )
    finally:
        try:
            for inp, state, before in zip(inputs, states, snapshots):
                assert _input_state(inp)[0] == state[0]
                assert torch.equal(_storage_bytes(inp).cpu(), before)
        except Exception as error:
            raise AssertionError("Compressed constructor mutated an input") from error
    try:
        assert isinstance(result, torch.Tensor)
        assert result.device == required_device and result.dtype == dtype
        assert result.layout == layout and tuple(result.shape) == tuple(expected_shape)
        names = (
            ("crow_indices", "col_indices", "values")
            if layout in (torch.sparse_csr, torch.sparse_bsr)
            else ("ccol_indices", "row_indices", "values")
        )
        for name, (part_dtype, shape) in zip(names, parts):
            _check_dense(getattr(result, name)(), part_dtype, shape, required_device)
    except Exception as error:
        raise AssertionError("Invalid compressed candidate output ABI") from error
    return result


def _assert_dense_equal(actual, expected, *, required_device, reference_device):
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    assert actual.device == required_device and expected.device == reference_device
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.layout == expected.layout == torch.strided
    actual, expected = _copy_tensor(actual, "cpu"), _copy_tensor(expected, "cpu")
    if actual.dtype in _FP8_DTYPES:
        actual, expected = actual.float(), expected.float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def _make_input(
    layout,
    shape,
    nnz,
    dtype,
    index_dtype=torch.int64,
    value_range=None,
    *,
    required_device=None,
):
    # Build sorted unique index entries in each compressed segment.
    if value_range is None:
        value_range = ["-1", "1"]
    device = _required_devices()[0] if required_device is None else required_device
    batch = shape[:-2]
    nrows, ncols = shape[-2], shape[-1]
    if layout in _BLOCK_LAYOUTS:
        bs0 = bs1 = _BLOCK_SIZE
    else:
        bs0 = bs1 = 1
    nblocks0, nblocks1 = nrows // bs0, ncols // bs1
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        comp_dim, plain_dim = nblocks0, nblocks1
    else:  # csc / bsc
        comp_dim, plain_dim = nblocks1, nblocks0
    entries = batch + (nnz,)
    assert 0 <= nnz <= comp_dim * plain_dim
    counts = torch.full((comp_dim,), nnz // comp_dim, dtype=torch.long)
    counts[: nnz % comp_dim] += 1
    compressed = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    plain = torch.arange(nnz) - torch.repeat_interleave(compressed[:-1], counts)
    compressed = compressed.expand(batch + (comp_dim + 1,)).contiguous()
    plain = plain.expand(entries).contiguous()
    block_shape = (bs0, bs1) if bs0 > 1 else ()
    values = _make_values(
        dtype,
        entries + block_shape,
        _range_for_dtype(dtype, value_range),
        required_device=device,
    )
    compressed = compressed.to(device=device, dtype=index_dtype)
    plain = plain.to(device=device, dtype=index_dtype)
    _check_dense(compressed, index_dtype, batch + (comp_dim + 1,), device)
    _check_dense(plain, index_dtype, entries, device)
    return compressed, plain, values


def _assert_result(
    res_out,
    ref_out,
    dtype,
    layout,
    index_dtype,
    *,
    required_device,
    reference_device,
    expected_shape,
):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.device == required_device and ref_out.device == reference_device
    assert ref_out.layout == layout and ref_out.dtype == dtype
    assert tuple(ref_out.shape) == tuple(expected_shape)
    assert res_out.layout == layout
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    assert tuple(res_out.shape) == tuple(expected_shape)
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out)

    if layout in (torch.sparse_csr, torch.sparse_bsr):
        res_c = torch.ops.aten.crow_indices(res_out)
        ref_c = torch.ops.aten.crow_indices(ref_out)
    else:
        res_c = torch.ops.aten.ccol_indices(res_out)
        ref_c = torch.ops.aten.ccol_indices(ref_out)
    if layout in (torch.sparse_csr, torch.sparse_bsr):
        res_p = torch.ops.aten.col_indices(res_out)
        ref_p = torch.ops.aten.col_indices(ref_out)
    else:
        res_p = torch.ops.aten.row_indices(res_out)
        ref_p = torch.ops.aten.row_indices(ref_out)
    assert res_c.dtype == index_dtype
    assert res_p.dtype == index_dtype
    # Indices are exact integer data.
    for actual, expected in (
        (res_c, ref_c),
        (res_p, ref_p),
        (res_out.values(), ref_out.values()),
    ):
        _assert_dense_equal(
            actual,
            expected,
            required_device=required_device,
            reference_device=reference_device,
        )


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize(
    "shape", [shape for shape in tu.selected_shapes() if len(shape) >= 2]
)
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in [torch.float32, torch.int32, torch.float8_e4m3fn]],
)
def test_sparse_compressed_tensor_shape_levels(shape, dtype):
    required_device, reference_device = _required_devices()
    layout = torch.sparse_csr
    nnz = 8
    compressed, plain, values = _make_input(
        layout, shape, nnz, dtype, required_device=required_device
    )
    ref_compressed = _to_reference(
        compressed, required_device=required_device, reference_device=reference_device
    )
    ref_plain = _to_reference(
        plain, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = _call_candidate(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
        required_device=required_device,
        expected_shape=shape,
    )

    assert res_out.shape == ref_out.shape == tuple(shape)
    _assert_result(
        res_out,
        ref_out,
        dtype,
        layout,
        torch.int64,
        required_device=required_device,
        reference_device=reference_device,
        expected_shape=shape,
    )


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _LAYOUT_CASES)
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in _SPARSE_COMPRESSED_DTYPES],
)
def test_sparse_compressed_tensor(case, dtype):
    required_device, reference_device = _required_devices()
    layout, shape, nnz, index_dtype = case
    compressed, plain, values = _make_input(
        layout, shape, nnz, dtype, index_dtype, required_device=required_device
    )
    ref_compressed = _to_reference(
        compressed, required_device=required_device, reference_device=reference_device
    )
    ref_plain = _to_reference(
        plain, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = _call_candidate(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
        required_device=required_device,
        expected_shape=shape,
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        layout,
        index_dtype,
        required_device=required_device,
        reference_device=reference_device,
        expected_shape=shape,
    )


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _NO_SIZE_CASES)
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in _SPARSE_COMPRESSED_DTYPES],
)
def test_sparse_compressed_tensor_no_size(case, dtype):
    required_device, reference_device = _required_devices()
    layout, shape, nnz, index_dtype = case
    compressed, plain, values = _make_input(
        layout, shape, nnz, dtype, index_dtype, required_device=required_device
    )
    ref_compressed = _to_reference(
        compressed, required_device=required_device, reference_device=reference_device
    )
    ref_plain = _to_reference(
        plain, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    expected_shape = tuple(ref_out.shape)
    res_out = _call_candidate(
        compressed,
        plain,
        values,
        dtype=dtype,
        layout=layout,
        device=compressed.device,
        required_device=required_device,
        expected_shape=expected_shape,
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        layout,
        index_dtype,
        required_device=required_device,
        reference_device=reference_device,
        expected_shape=expected_shape,
    )


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _EMPTY_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_COMPRESSED_DTYPES)
def test_sparse_compressed_tensor_empty(case, dtype):
    required_device, reference_device = _required_devices()
    layout, shape = case
    compressed, plain, values = _make_input(
        layout, shape, 0, dtype, required_device=required_device
    )
    ref_compressed = _to_reference(
        compressed, required_device=required_device, reference_device=reference_device
    )
    ref_plain = _to_reference(
        plain, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = _call_candidate(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
        required_device=required_device,
        expected_shape=shape,
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        layout,
        torch.int64,
        required_device=required_device,
        reference_device=reference_device,
        expected_shape=shape,
    )


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("case", _VALUE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in _SPARSE_COMPRESSED_DTYPES],
)
def test_sparse_compressed_tensor_value_ranges(case, value_range, dtype):
    required_device, reference_device = _required_devices()
    layout, shape, nnz, index_dtype = case
    compressed, plain, values = _make_input(
        layout,
        shape,
        nnz,
        dtype,
        index_dtype,
        value_range,
        required_device=required_device,
    )
    ref_compressed = _to_reference(
        compressed, required_device=required_device, reference_device=reference_device
    )
    ref_plain = _to_reference(
        plain, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = _call_candidate(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
        required_device=required_device,
        expected_shape=shape,
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        layout,
        index_dtype,
        required_device=required_device,
        reference_device=reference_device,
        expected_shape=shape,
    )


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize(
    "dtype, scenario",
    [
        (dtype, scenario)
        for dtype, scenario in tu.selected_cases(
            tu.special_value_cases(_SPARSE_COMPRESSED_DTYPES)
        )
    ],
)
def test_sparse_compressed_tensor_nan_inf_values(dtype, scenario):
    required_device, reference_device = _required_devices()
    layout = torch.sparse_csr
    shape = (3, 4)
    crow_t = torch.tensor([0, 2, 4, 7], dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(
        [0, 1, 0, 2, 0, 1, 2], dtype=torch.long, device=flag_gems.device
    )
    _check_dense(crow_t, torch.int64, (4,), required_device)
    _check_dense(col_t, torch.int64, (7,), required_device)
    values = _make_special_values(dtype, scenario, required_device=required_device)
    ref_crow = _to_reference(
        crow_t, required_device=required_device, reference_device=reference_device
    )
    ref_col = _to_reference(
        col_t, required_device=required_device, reference_device=reference_device
    )
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_crow,
        ref_col,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_crow.device,
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=crow_t.device,
        required_device=required_device,
        expected_shape=shape,
    )

    assert res_out.layout == layout
    assert res_out.shape == ref_out.shape == shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert torch.ops.aten._nnz(res_out) == torch.ops.aten._nnz(ref_out) == 7
    _assert_result(
        res_out,
        ref_out,
        dtype,
        layout,
        torch.int64,
        required_device=required_device,
        reference_device=reference_device,
        expected_shape=shape,
    )


@pytest.mark.sparse_compressed_tensor
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="Kunlunxin stored-values backward requires unsupported native index_add",
)
def test_sparse_compressed_tensor_backward(dtype):
    required_device, reference_device = _required_devices()
    layout, shape, nnz = torch.sparse_csr, (5, 4), 7
    compressed, plain, values = _make_input(
        layout, shape, nnz, dtype, required_device=required_device
    )
    ref_compressed = _to_reference(
        compressed, required_device=required_device, reference_device=reference_device
    )
    ref_plain = _to_reference(
        plain, required_device=required_device, reference_device=reference_device
    )
    values.requires_grad_(True)
    values_shape = tuple(values.shape)
    backward_inputs = (compressed, plain, values)
    backward_states = [_input_state(inp) for inp in backward_inputs]
    backward_bytes = [_storage_bytes(inp).cpu().clone() for inp in backward_inputs]
    ref_values = _to_reference(
        values, required_device=required_device, reference_device=reference_device
    )

    ref_out = torch.ops.aten.sparse_compressed_tensor(
        ref_compressed,
        ref_plain,
        ref_values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=ref_compressed.device,
    )
    res_out = _call_candidate(
        compressed,
        plain,
        values,
        list(shape),
        dtype=dtype,
        layout=layout,
        device=compressed.device,
        required_device=required_device,
        expected_shape=shape,
    )

    _assert_result(
        res_out,
        ref_out,
        dtype,
        layout,
        torch.int64,
        required_device=required_device,
        reference_device=reference_device,
        expected_shape=shape,
    )

    weights = torch.linspace(-1.0, 1.0, 7, dtype=dtype, device=required_device)
    ref_weights = _to_reference(
        weights, required_device=required_device, reference_device=reference_device
    )
    grad_ref = torch.autograd.grad((ref_out.values() * ref_weights).sum(), ref_values)[
        0
    ]
    grad_res = torch.autograd.grad((res_out.values() * weights).sum(), values)[0]
    for inp, state, before in zip(backward_inputs, backward_states, backward_bytes):
        assert _input_state(inp)[0] == state[0]
        assert torch.equal(_storage_bytes(inp).cpu(), before)
    _check_dense(grad_res, dtype, values_shape, required_device)
    _check_dense(grad_ref, dtype, values_shape, reference_device)
    utils.gems_assert_close(
        _copy_tensor(grad_res, "cpu"), _copy_tensor(grad_ref, "cpu"), dtype
    )


def _negative_csr_inputs(dtype=torch.float32):
    layout, shape, nnz = torch.sparse_csr, (5, 4), 7
    compressed, plain, values = _make_input(layout, shape, nnz, dtype)
    return compressed, plain, values, shape


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_missing_layout():
    compressed, plain, values, shape = _negative_csr_inputs()
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            dtype=torch.float32,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.sparse_compressed_tensor(
            compressed,
            plain,
            values,
            list(shape),
            dtype=torch.float32,
            device=compressed.device,
        )


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_coo_layout():
    compressed, plain, values, shape = _negative_csr_inputs()
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.sparse_compressed_tensor(
            compressed,
            plain,
            values,
            list(shape),
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=compressed.device,
        )


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_dtype_mismatch():
    compressed, plain, values, shape = _negative_csr_inputs()
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            dtype=torch.float64,
            layout=torch.sparse_csr,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.sparse_compressed_tensor(
            compressed,
            plain,
            values,
            list(shape),
            dtype=torch.float64,
            layout=torch.sparse_csr,
            device=compressed.device,
        )


@pytest.mark.sparse_compressed_tensor
def test_sparse_compressed_tensor_rejects_non_float32_values_without_dtype():
    compressed, plain, values, shape = _negative_csr_inputs(torch.bfloat16)
    ref_compressed = tu.to_reference(compressed)
    ref_plain = tu.to_reference(plain)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_compressed_tensor(
            ref_compressed,
            ref_plain,
            ref_values,
            list(shape),
            layout=torch.sparse_csr,
            device=ref_compressed.device,
        )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.sparse_compressed_tensor(
            compressed,
            plain,
            values,
            list(shape),
            layout=torch.sparse_csr,
            device=compressed.device,
        )
