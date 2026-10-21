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


def _check_input(inp, dtype, shape):
    assert inp.dtype == dtype and inp.shape == torch.Size(shape)
    assert inp.device == _device(flag_gems.device)
    return inp


def _cpu_input(dtype, shape, value_range):
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


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        inp = _copy_to(_cpu_input(dtype, shape, value_range), flag_gems.device)
    else:
        inp = tu.make_input(dtype, shape, value_range)
    return _check_input(inp, dtype, shape)


def _make_special_input(dtype, scenario):
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
    return _check_input(inp, dtype, (5,))


def _repeat(inp, *sizes):
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


# _dimV reports the dense dimension count of a COO tensor. Dense and
# compressed sparse inputs are rejected.
_DIMV_DTYPES = (
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)

# COO: (shape, dense_dim).
_DIMV_COO_CASES = tu.selected_cases(
    [
        ((5,), 0),
        ((3, 4), 0),
        ((3, 4), 1),
        ((8, 8, 8), 0),
        ((3, 4, 2), 1),
        ((3, 4, 2), 2),
        ((4, 3, 4, 5), 3),
        ((3, 4, 5, 4, 5), 2),
        ((12, 9, 3, 6), 0),
        ((3, 6, 4, 4, 6, 5), 2),
        ((7, 3, 12, 4, 2, 15), 3),
        ((3, 4, 2, 5, 3, 4, 2), 4),
        ((2, 4, 2, 4, 2, 4), 3),
    ],
    quick=[
        ((2, 19, 7), 0),
        ((2, 19, 7), 1),
        ((2, 19, 7), 2),
        ((2, 19, 7, 5), 0),
        ((2, 19, 7, 5), 1),
        ((2, 19, 7, 5), 3),
        ((2, 19, 7, 5, 3), 0),
        ((2, 19, 7, 5, 3), 2),
    ],
)

# Value ranges: (shape, dense_dim) for hybrid COO layouts.
_DIMV_HYBRID_CASES = tu.selected_cases(
    [
        ((3, 4), 1),
        ((3, 4, 2), 1),
        ((4, 3, 4, 5), 3),
        ((3, 4, 5, 4, 5), 2),
    ],
    quick=[((2, 19, 7), 1), ((2, 19, 7), 2), ((2, 19, 7, 5), 1), ((2, 19, 7, 5, 3), 1)],
)

# Number of dense trailing dimensions for each shared shape.
_SHAPE_DENSE_DIM = {
    (1,): 0,
    (256,): 0,
    (1024, 1024): 1,
    (20, 320, 15): 1,
    (16, 128, 64, 60): 2,
    (16, 7, 57, 32, 29): 3,
}
_DIMV_SHAPE_CASES = [
    (tuple(shape), _SHAPE_DENSE_DIM.get(tuple(shape), 0))
    for shape in tu.selected_shapes()
    if len(shape) >= 1
]


def _make_coo_input(shape, dense_dim, dtype, value_range, nnz=8, seed=0):
    """Build COO with the requested dense suffix and seeded, possibly duplicate indices."""
    shape = tuple(shape)
    if not 0 <= dense_dim < len(shape):
        raise ValueError("dense_dim must leave at least one sparse dimension")
    sparse_dim = len(shape) - dense_dim
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = _make_input(dtype, (nnz,) + tuple(dense_shape), value_range)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_empty_coo(shape, dense_dim, dtype):
    shape = tuple(shape)
    sparse_dim = len(shape) - dense_dim
    indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
    values = torch.empty(
        (0,) + shape[sparse_dim:], dtype=dtype, device=flag_gems.device
    )
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _assert_result(res_out, ref_out):
    assert isinstance(res_out, int) and not isinstance(res_out, bool)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_dimV
@pytest.mark.parametrize("case", _DIMV_COO_CASES)
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_coo(case, dtype):
    shape, dense_dim = case
    inp = _make_coo_input(shape, dense_dim, dtype, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = flag_gems._dimV(inp)

    _assert_result(res_out, ref_out)
    # Pure metadata query: the input layout is untouched.
    assert inp.dense_dim() == dense_dim
    assert inp.sparse_dim() == len(shape) - dense_dim
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark.underscore_dimV
@pytest.mark.parametrize("case", _DIMV_SHAPE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_shape_value_range_grid(case, value_range, dtype):
    shape, dense_dim = case
    inp = _make_coo_input(shape, dense_dim, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = flag_gems._dimV(inp)

    _assert_result(res_out, ref_out)
    assert inp.dense_dim() == dense_dim
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark.underscore_dimV
@pytest.mark.parametrize("case", _DIMV_HYBRID_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_hybrid_value_ranges(case, value_range, dtype):
    shape, dense_dim = case
    inp = _make_coo_input(shape, dense_dim, dtype, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = flag_gems._dimV(inp)

    _assert_result(res_out, ref_out)
    assert inp.dense_dim() > 0
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark.underscore_dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_empty(dtype):
    shape, dense_dim = (3, 4), 0
    inp = _make_empty_coo(shape, dense_dim, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = flag_gems._dimV(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.underscore_dimV
@pytest.mark.parametrize("shape, dense_dim", [((4, 5, 6), 1), ((4, 5, 6), 2)])
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_empty_hybrid(shape, dense_dim, dtype):
    inp = _make_empty_coo(shape, dense_dim, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = flag_gems._dimV(inp)

    _assert_result(res_out, ref_out)
    assert inp.dense_dim() == dense_dim
    assert inp.sparse_dim() + inp.dense_dim() == len(shape)


@pytest.mark.underscore_dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_single_entry(dtype):
    shape, dense_dim = (3, 4, 5), 2
    inp = _make_coo_input(shape, dense_dim, dtype, ["-1", "1"], nnz=1)
    assert inp._nnz() == 1
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = flag_gems._dimV(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.underscore_dimV
@pytest.mark.parametrize("dtype", _DIMV_DTYPES)
def test__dimV_uncoalesced(dtype):
    shape = (3, 4)
    indices = torch.tensor([[0, 0, 1, 2, 0]], dtype=torch.long)
    values = _make_input(dtype, (5, 4), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = flag_gems._dimV(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.underscore_dimV
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DIMV_DTYPES))
)
def test__dimV_nan_inf_values_ignored(dtype, scenario):
    values = _repeat(_make_special_input(dtype, scenario), 3)[:12].reshape(6, 2)
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6, 2), device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._dimV(ref_inp)
    res_out = flag_gems._dimV(inp)

    _assert_result(res_out, ref_out)


_NEGATIVE_EXC = (
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    IndexError,
)


@pytest.mark.underscore_dimV
def test__dimV_dense_raises():
    inp = _make_input(torch.float32, (4, 4), ["-1", "1"])
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimV(tu.to_reference(inp))
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._dimV(inp)


@pytest.mark.underscore_dimV
def test__dimV_csr_raises():
    crow_indices = torch.tensor([0, 1, 2])
    col_indices = torch.tensor([0, 1])
    values = _make_input(torch.float32, (2,), ["-1", "1"])
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (2, 3), device=flag_gems.device
    )
    with pytest.raises(NotImplementedError):
        torch.ops.aten._dimV(tu.to_reference(inp))
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._dimV(inp)


@pytest.mark.underscore_dimV
def test__dimV_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._dimV(3.14)
    with pytest.raises(_NEGATIVE_EXC):
        flag_gems._dimV(3.14)
