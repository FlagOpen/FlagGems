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
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
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


def _assert_output_equal(actual, expected):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and actual.dtype in _FP8_DTYPES:
        _assert_values_equal(actual, expected, equal_nan=True)
    else:
        tu.assert_result_equal(actual, expected)


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# _values returns a non-differentiable view of the COO value storage.
# Explicit zeros and duplicate entries are preserved; dense and CSR inputs fail.
_VALUES_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

_VALUES_QUICK_CASES = [((2, 19, 7), 2, 8), ((3, 4), 2, 7), ((3, 4, 2), 2, 12)]

# COO: (shape, sparse_dim, nnz).
_VALUES_COO_CASES = tu.selected_cases(
    [
        ((5,), 1, 4),
        ((3, 4), 2, 7),
        ((3, 4), 1, 16),
        ((8, 8, 8), 3, 32),
        ((3, 4, 2), 2, 12),
        ((4, 3, 4, 5), 1, 24),
        ((3, 4, 5, 4, 5), 3, 40),
        ((12, 9, 3, 6), 4, 48),
        ((3, 6, 4, 4, 6, 5), 4, 64),
        ((7, 3, 12, 4, 2, 15), 5, 80),
        ((3, 4, 2, 5, 3, 4, 2), 3, 96),
    ],
    quick=_VALUES_QUICK_CASES,
)

# Value-range layouts: all-sparse and hybrid COO.
_VALUES_RANGE_CASES = tu.selected_cases(
    [
        ((3, 4), 2, 7),
        ((3, 4, 2), 2, 12),
        ((12, 9, 3, 6), 4, 48),
    ],
    quick=_VALUES_QUICK_CASES,
)


def _make_coo_input(shape, sparse_dim, nnz, dtype, value_range, seed=0):
    # Seeded CPU indices may repeat; values are created on the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = _make_values(dtype, (nnz,) + dense_shape, value_range)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _assert_result(res_out, ref_out, inp, ref_inp):
    assert inp.dtype == ref_inp.dtype and inp.shape == ref_inp.shape
    assert inp.layout == ref_inp.layout and inp.device == _device(flag_gems.device)
    assert ref_inp.device == (torch.device("cpu") if utils.TO_CPU else inp.device)
    assert res_out.dtype == ref_out.dtype and res_out.shape == ref_out.shape
    assert res_out.layout == ref_out.layout == torch.strided
    assert res_out.device == inp.device
    # Check exact output, storage aliasing and unchanged input values.
    _assert_output_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp._values().data_ptr()
    _assert_values_equal(inp._values(), ref_inp._values(), equal_nan=True)


@pytest.mark.underscore_values
@pytest.mark.parametrize("case", _VALUES_COO_CASES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
        )
        for dtype in _VALUES_DTYPES
    ],
)
def test__values_layouts(case, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_values
@pytest.mark.parametrize("case", _VALUES_RANGE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
        )
        for dtype in _VALUES_DTYPES
    ],
)
def test__values_value_ranges(case, value_range, dtype):
    shape, sparse_dim, nnz = case
    inp = _make_coo_input(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_values
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
        )
        for dtype in _VALUES_DTYPES
    ],
)
def test__values_empty(dtype):
    shape, sparse_dim = (3, 4), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == 0
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_values
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
        )
        for dtype in _VALUES_DTYPES
    ],
)
def test__values_empty_hybrid(dtype):
    shape, sparse_dim = (4, 5, 6), 2
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(0, 6, dtype=dtype, device=flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == 0
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_values
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
        )
        for dtype in _VALUES_DTYPES
    ],
)
def test__values_full_storage(dtype):
    shape = (2, 3)
    nnz = shape[0] * shape[1]
    indices = torch.stack(
        torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    ).reshape(2, nnz)
    values = _make_values(dtype, (nnz,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert inp._nnz() == nnz
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_values
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
        )
        for dtype in _VALUES_DTYPES
    ],
)
def test__values_uncoalesced(dtype):
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 1, 2, 3, 1]], dtype=torch.long)
    values = _make_values(dtype, (5,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_values
@pytest.mark.parametrize(
    "dtype, scenario",
    [
        pytest.param(
            dtype,
            scenario,
        )
        for dtype, scenario in tu.selected_cases(tu.special_value_cases(_VALUES_DTYPES))
    ],
)
def test__values_nan_inf(dtype, scenario):
    values = _repeat_values(_make_special_values(dtype, scenario), 2)[:6]
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._values(ref_inp)
    res_out = flag_gems._values(inp)

    _assert_result(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_values
def test__values_dense_raises():
    inp = _make_values(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._values(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems._values(inp)


@pytest.mark.underscore_values
def test__values_csr_raises():
    crow_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    col_indices = torch.tensor([0, 1], dtype=torch.long)
    values = _make_values(torch.float32, (2,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 3), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._values(tu.to_reference(inp))
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        flag_gems._values(inp)


@pytest.mark.underscore_values
def test__values_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._values(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._values(3.14)
    with pytest.raises(RuntimeError):
        torch.ops.aten._values("not-a-tensor")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._values("not-a-tensor")
