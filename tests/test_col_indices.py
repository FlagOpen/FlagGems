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


def _device(device):
    expected = torch.device(device)
    if expected.index is None and expected.type not in ("cpu", "meta"):
        expected = torch.device(
            expected.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return expected


def _view_from_storage(inp, storage, device):
    out = torch.empty(0, dtype=inp.dtype, device=device).set_(
        storage, inp.storage_offset(), inp.size(), inp.stride()
    )
    if inp.is_conj():
        out = out.conj()
    if inp.is_neg():
        out = torch._neg_view(out)
    assert out.dtype == inp.dtype and out.device == _device(device)
    assert out.shape == inp.shape and out.stride() == inp.stride()
    assert out.storage_offset() == inp.storage_offset()
    assert (out.is_conj(), out.is_neg()) == (inp.is_conj(), inp.is_neg())
    return out.requires_grad_(inp.requires_grad)


def _copy_to(inp, device):
    # Copy physical bytes before restoring the view and lazy flags. A typed
    # cross-device copy can resolve conjugation or require unavailable kernels.
    nbytes = inp.untyped_storage().nbytes()
    raw = torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        inp.untyped_storage(), 0, (nbytes,), (1,)
    )
    raw = raw.to(device).clone()
    return _view_from_storage(inp, raw.untyped_storage(), raw.device)


def _check_values(inp, dtype, shape):
    assert inp.dtype == dtype and inp.shape == torch.Size(shape)
    assert inp.device == _device(flag_gems.device)
    return inp


def _cpu_values(dtype, shape, value_range):
    # The original factory recipe, with the same bounds and degenerate fills.
    low, high = (tu.resolve_bound(v, dtype) for v in value_range)
    if not dtype.is_floating_point:
        low, high = int(low), int(high)
        dtype_min, dtype_max = tu.dtype_bounds(dtype)
        low = max(low, int(dtype_min))
        high = min(high, int(dtype_max))
        low = min(low, high)
    if low == high:
        return torch.full(shape, low, dtype=dtype, device="cpu")
    return torch.testing.make_tensor(
        shape, dtype=dtype, device="cpu", low=low, high=high
    )


def _make_values(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        inp = _copy_to(_cpu_values(dtype, shape, value_range), flag_gems.device)
    else:
        inp = tu.make_input(dtype, shape, value_range)
    return _check_values(inp, dtype, shape)


def _make_special_values(dtype, scenario):
    if (
        flag_gems.vendor_name in ("ascend", "kunlunxin")
        and dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    ) or (flag_gems.vendor_name == "ascend" and dtype == torch.float64):
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        value = torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu")
        inp = _copy_to(value.to(dtype), flag_gems.device)
    else:
        inp = tu.make_special_input(dtype, scenario)
    return _check_values(inp, dtype, (5,))


def _repeat_values(inp, *sizes):
    if (
        flag_gems.vendor_name in ("ascend", "kunlunxin")
        and inp.dtype
        in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        and inp.device.type != "cpu"
    ):
        return _copy_to(_copy_to(inp, "cpu").repeat(*sizes), inp.device)
    return inp.repeat(*sizes)


def _assert_values_equal(actual, expected, *, equal_nan=False):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and actual.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        # Retain the original tensor contract before observing FP8 values on
        # CPU, where decoding avoids the missing device comparison kernels.
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape and actual.layout == expected.layout
        assert actual.device == _device(flag_gems.device)
        assert expected.device == (
            torch.device("cpu") if utils.TO_CPU else actual.device
        )
        actual = _copy_to(actual, "cpu").float()
        expected = _copy_to(expected, "cpu").float()
    utils.gems_assert_equal(actual, expected, equal_nan=equal_nan)


# col_indices returns a view of the column index storage in CSR/BSR tensors.
_COL_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

# (layout, shape, nnz, block_shape), including batched CSR/BSR.
_COL_CASES = tu.selected_cases(
    [
        ("csr", (5, 4), 7, None),
        ("csr", (3, 8), 16, None),
        ("csr", (8, 3), 12, None),
        ("csr", (4, 4), 16, None),
        ("csr", (1, 6), 4, None),
        ("csr", (3, 5, 4), 7, None),
        ("csr", (2, 4, 6), 12, None),
        ("csr", (2, 3, 4, 5), 8, None),
        ("bsr", (4, 6), 4, (2, 2)),
        ("bsr", (8, 8), 6, (2, 2)),
        ("csr", (12, 9, 3, 6), 9, None),
        ("csr", (3, 6, 4, 4, 6, 5), 11, None),
        ("csr", (7, 3, 12, 4, 2, 15), 10, None),
        ("csr", (3, 4, 2, 5, 3, 4, 2), 8, None),
        ("bsr", (12, 12), 6, (3, 4)),
        ("bsr_batch", (2, 4, 6), 4, (2, 2)),
        ("bsr_batch", (2, 8, 12), 6, (4, 4)),
    ],
    quick=[("csr", (2, 19, 7), 8, None)],
)

# Value-range layouts: plain/batched CSR and block storage.
_COL_RANGE_CASES = tu.selected_cases(
    [
        ("csr", (5, 4), 7, None),
        ("csr", (3, 5, 4), 7, None),
        ("bsr", (4, 6), 4, (2, 2)),
    ],
    quick=[("csr", (2, 19, 7), 8, None)],
)

# (layout, matrix_shape, values_shape, compressed_indices, plain_indices).
_INDEX_LAYOUT_CASES = [
    (torch.sparse_csr, (4, 6), (4,), [0, 2, 2, 3, 4], [0, 4, 1, 5]),
    (torch.sparse_bsr, (4, 6), (3, 2, 3), [0, 2, 3], [0, 1, 1]),
    (torch.sparse_csr, (4, 6), (0,), [0, 0, 0, 0, 0], []),
    (torch.sparse_bsr, (4, 6), (0, 2, 3), [0, 0, 0], []),
]
_INDEX_CASES = tu.selected_cases(
    [
        (case, batch_shape, dense_shape)
        for case in _INDEX_LAYOUT_CASES
        for batch_shape in [(), (2,), (2, 3)]
        for dense_shape in [(), (2,)]
    ],
    quick=[
        (_INDEX_LAYOUT_CASES[0], (), ()),
        (_INDEX_LAYOUT_CASES[1], (2,), (2,)),
        (_INDEX_LAYOUT_CASES[2], (), ()),
        (_INDEX_LAYOUT_CASES[3], (2,), ()),
    ],
)


def _make_compressed_indices(batch, n_rows, n_cols, nnz):
    """Build sorted CSR indices with a row sentinel; each coordinate occurs once."""
    entries = tuple(batch) + (nnz,)
    assert 0 <= nnz <= n_rows * n_cols
    positions = torch.arange(nnz, dtype=torch.long) * (n_rows * n_cols) // max(nnz, 1)
    rows = (positions // n_cols).expand(entries).contiguous()
    cols = (positions % n_cols).expand(entries).contiguous()
    counts = torch.zeros(tuple(batch) + (n_rows,), dtype=torch.long)
    if nnz > 0:
        counts.scatter_add_(-1, rows, torch.ones(entries, dtype=torch.long))
    crow = torch.zeros(tuple(batch) + (n_rows + 1,), dtype=torch.long)
    crow[..., 1:] = counts.cumsum(-1)
    return crow, cols


def _build_csr(shape, nnz, dtype, value_range):
    batch, n_rows, n_cols = shape[:-2], shape[-2], shape[-1]
    crow, cols = _make_compressed_indices(batch, n_rows, n_cols, nnz)
    values = _make_values(dtype, tuple(batch) + (nnz,), value_range)
    return torch.sparse_csr_tensor(
        crow.to(flag_gems.device),
        cols.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _build_bsr(shape, nnz, blocks, dtype, value_range):
    batch, n_rows, n_cols = shape[:-2], shape[-2], shape[-1]
    block_rows, block_cols = blocks
    n_row_blocks = n_rows // block_rows
    n_col_blocks = n_cols // block_cols
    crow, cols = _make_compressed_indices(batch, n_row_blocks, n_col_blocks, nnz)
    values = _make_values(
        dtype, tuple(batch) + (nnz, block_rows, block_cols), value_range
    )
    return torch.sparse_bsr_tensor(
        crow.to(flag_gems.device),
        cols.to(flag_gems.device),
        values.to(flag_gems.device),
        shape,
    )


def _build_input(layout, shape, nnz, blocks, dtype, value_range=("-1", "1")):
    if layout == "csr":
        return _build_csr(shape, nnz, dtype, value_range)
    if layout in ("bsr", "bsr_batch"):
        return _build_bsr(shape, nnz, blocks, dtype, value_range)
    raise ValueError(f"unknown layout {layout}")


def _assert_result(res_out, ref_out, inp, ref_inp):
    assert inp.dtype == ref_inp.dtype and inp.shape == ref_inp.shape
    assert inp.layout == ref_inp.layout and inp.device == _device(flag_gems.device)
    assert ref_inp.device == (torch.device("cpu") if utils.TO_CPU else inp.device)
    assert res_out.dtype == ref_out.dtype and res_out.shape == ref_out.shape
    assert res_out.layout == ref_out.layout == torch.strided
    assert res_out.device == inp.device
    # Check exact output, storage aliasing and unchanged input metadata/values.
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp.col_indices().data_ptr()
    utils.gems_assert_equal(inp.crow_indices(), ref_inp.crow_indices())
    utils.gems_assert_equal(inp.col_indices(), ref_inp.col_indices())
    if inp.dtype.is_floating_point or inp.dtype.is_complex:
        _assert_values_equal(inp.values(), ref_inp.values(), equal_nan=True)
    else:
        _assert_values_equal(inp.values(), ref_inp.values())


def _dtype_marks(dtype):
    return pytest.mark.skipif(
        flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64,
        reason="Kunlunxin float64 allocation and transfer narrow to float32",
    )


_COL_INPUT_DTYPE_CASES = [
    pytest.param(dtype, marks=_dtype_marks(dtype)) for dtype in _COL_DTYPES
]


@pytest.mark.col_indices
@pytest.mark.parametrize("case,batch_shape,dense_shape", _INDEX_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", _COL_INPUT_DTYPE_CASES)
def test_col_indices_index_layouts(case, batch_shape, dense_shape, dtype, index_dtype):
    layout, matrix_shape, values_shape, compressed, plain = case
    compressed = torch.tensor(compressed, dtype=index_dtype, device=flag_gems.device)
    plain = torch.tensor(plain, dtype=index_dtype, device=flag_gems.device)
    compressed = compressed.expand(batch_shape + compressed.shape).contiguous()
    plain = plain.expand(batch_shape + plain.shape).contiguous()
    values = _make_values(dtype, batch_shape + values_shape + dense_shape, ["-1", "1"])
    inp = torch.sparse_compressed_tensor(
        compressed,
        plain,
        values,
        size=batch_shape + matrix_shape + dense_shape,
        layout=layout,
        check_invariants=True,
    )
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)
    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("case", _COL_CASES)
@pytest.mark.parametrize("dtype", _COL_INPUT_DTYPE_CASES)
def test_col_indices_layouts(case, dtype):
    layout, shape, nnz, blocks = case
    inp = _build_input(layout, shape, nnz, blocks, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("case", _COL_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _COL_INPUT_DTYPE_CASES)
def test_col_indices_value_ranges(case, value_range, dtype):
    layout, shape, nnz, blocks = case
    inp = _build_input(layout, shape, nnz, blocks, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_INPUT_DTYPE_CASES)
def test_col_indices_empty(dtype):
    # Empty tensors have null data pointers, so pointer equality cannot prove aliasing.
    inp = _build_input("csr", (4, 5), 0, None, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_INPUT_DTYPE_CASES)
def test_col_indices_empty_batched(dtype):
    inp = _build_input("csr", (2, 4, 5), 0, None, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_INPUT_DTYPE_CASES)
def test_col_indices_empty_bsr(dtype):
    inp = _build_input("bsr", (4, 6), 0, (2, 2), dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_INPUT_DTYPE_CASES)
def test_col_indices_single_row(dtype):
    inp = _build_input("csr", (1, 7), 5, None, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_INPUT_DTYPE_CASES)
def test_col_indices_unchecked_uncoalesced(dtype):
    shape = (4, 3)
    crow = torch.tensor([0, 3, 3, 5, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 0, 2, 1, 2], dtype=torch.long, device=flag_gems.device)
    assert cols[0].item() == cols[1].item()
    values = _make_values(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_INPUT_DTYPE_CASES)
def test_col_indices_full_storage(dtype):
    shape = (2, 3)
    crow = torch.tensor([0, 3, 6], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long, device=flag_gems.device)
    values = _make_values(dtype, (6,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(crow, cols, values.to(flag_gems.device), shape)
    assert inp._nnz() == 6
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize("dtype", _COL_INPUT_DTYPE_CASES)
def test_col_indices_bsr_rectangular_blocks(dtype):
    inp = _build_input("bsr", (12, 12), 6, (3, 4), dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
            marks=_dtype_marks(dtype),
        )
        for dtype, scenario in tu.selected_cases(tu.special_value_cases(_COL_DTYPES))
    ],
)
def test_col_indices_nan_inf_values_ignored(dtype, scenario):
    shape = (3, 4)
    crow = torch.tensor([0, 2, 4, 5], dtype=torch.long, device=flag_gems.device)
    cols = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long, device=flag_gems.device)
    values = _make_special_values(dtype, scenario)
    inp = torch.sparse_csr_tensor(crow, cols, values, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.col_indices(ref_inp)
    res_out = flag_gems.col_indices(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.col_indices
def test_col_indices_dense_raises():
    inp = _make_values(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.col_indices(inp)


@pytest.mark.col_indices
def test_col_indices_csc_raises():
    ccol_indices = torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=flag_gems.device)
    values = _make_values(torch.float32, (4,), ["-1", "1"])
    inp = torch.sparse_csc_tensor(
        ccol_indices, row_indices, values.to(flag_gems.device), (4, 2)
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.col_indices(inp)


@pytest.mark.col_indices
def test_col_indices_bsc_raises():
    ccol_indices = torch.tensor([0, 1, 2], dtype=torch.long, device=flag_gems.device)
    row_indices = torch.tensor([0, 1], dtype=torch.long, device=flag_gems.device)
    values = torch.randn(2, 2, 2, dtype=torch.float32, device=flag_gems.device)
    inp = torch.sparse_bsc_tensor(
        ccol_indices, row_indices, values, (4, 4), device=flag_gems.device
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.col_indices(inp)


@pytest.mark.col_indices
def test_col_indices_coo_raises():
    inp = torch.randn(3, 4, device=flag_gems.device).to_sparse_coo()
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices(tu.to_reference(inp))
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.col_indices(inp)


@pytest.mark.col_indices
def test_col_indices_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.col_indices(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.col_indices(3.14)
