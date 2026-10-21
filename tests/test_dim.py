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


def _batch_values(values, count):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and values.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        host = _copy_to(values, "cpu").expand(count, -1).contiguous()
        return _copy_to(host, values.device)
    return values.expand(count, -1).contiguous()


def _to_reference(inp):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and utils.TO_CPU
        and inp.device.type != "cpu"
        and inp.layout == torch.strided
        and inp.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and not inp.is_contiguous()
    ):
        return _copy_to(inp, "cpu")
    return tu.to_reference(inp)


# dim returns the logical rank for dense, COO and CSR layouts.
_DIM_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

# Dense shapes, including scalars and higher ranks.
_DENSE_SHAPES = tu.selected_cases(
    [
        (),
        (5,),
        (3, 4),
        (8, 8, 8),
        (3, 4, 2, 5),
        (3, 4, 5, 4, 5),
        (3, 6, 4, 4, 6, 5, 4),
        (7, 3, 12, 4, 2, 15, 2, 2),
    ],
    quick=[(2, 19, 7)],
)

# COO: (sparse_shape, dense_shape, nnz).
_COO_CASES = tu.selected_cases(
    [
        ((4, 4), (), 8),
        ((8, 8, 8), (), 64),
        ((4, 4), (3,), 8),
        ((2, 3, 4), (5,), 12),
        ((16, 16), (7, 13), 40),
        ((2, 3, 4), (5, 6), 12),
        ((3,), (4, 5, 6), 2),
        ((12, 9, 3, 6), (4,), 9),
        ((3, 4, 2, 5, 3), (4, 2), 11),
    ],
    quick=[((2, 19, 7), (), 8)],
)

# COO value ranges: all-sparse and hybrid layouts.
_COO_RANGE_CASES = tu.selected_cases(
    [
        ((3, 4), (), 7),
        ((3, 4), (3,), 8),
        ((12, 9, 3, 6), (4,), 9),
    ],
    quick=[((2, 19, 7), (), 8)],
)

# CSR: (shape, nnz), including batched tensors.
_CSR_CASES = tu.selected_cases(
    [
        ((4, 4), 3),
        ((2, 4, 4), 5),
        ((3, 5, 7), 3),
        ((3, 4, 4), 4),
    ],
    quick=[((2, 19, 7), 3)],
)

# CSR value ranges: plain and batched layouts.
_CSR_RANGE_CASES = tu.selected_cases(
    [
        ((4, 4), 3),
        ((2, 4, 4), 5),
    ],
    quick=[((2, 19, 7), 3)],
)


def _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range, seed=0):
    # Seeded CPU indices allow duplicates; values are created on the test device.
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = _make_input(dtype, (nnz,) + tuple(dense_shape), value_range)
    size = tuple(sparse_shape) + tuple(dense_shape)
    return torch.sparse_coo_tensor(indices, values, size, device=flag_gems.device)


def _make_csr(shape, nnz, dtype, value_range):
    if len(shape) == 2:
        rows, cols = shape
    else:
        _, rows, cols = shape
    assert 0 <= nnz <= rows * cols
    counts = torch.full((rows,), nnz // rows, dtype=torch.long)
    counts[: nnz % rows] += 1
    crow_indices = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    col_indices = torch.arange(nnz) - torch.repeat_interleave(crow_indices[:-1], counts)
    values = _make_input(dtype, (nnz,), value_range)
    if len(shape) == 3:
        # Batched CSR: every batch stores the same nnz entries (shared
        # crow/col pattern), so the logical rank is 3.
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = _batch_values(values, shape[0])
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _make_empty_csr(shape, dtype):
    """Build an empty plain or batched CSR tensor."""
    if len(shape) == 2:
        rows, _ = shape
        crow_indices = torch.zeros(rows + 1, dtype=torch.long)
        col_indices = torch.empty(0, dtype=torch.long)
        values = torch.empty(0, dtype=dtype, device=flag_gems.device)
    else:
        _, rows, _ = shape
        crow_indices = torch.zeros(shape[0], rows + 1, dtype=torch.long)
        col_indices = torch.empty(shape[0], 0, dtype=torch.long)
        values = torch.empty(shape[0], 0, dtype=dtype, device=flag_gems.device)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=flag_gems.device
    )


def _assert_result(res_out, ref_out):
    assert type(res_out) is int
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize("shape", _DENSE_SHAPES)
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_dense_layouts(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize("shape", [(0,), (0, 5), (2, 0, 3)])
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_empty_dense(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    assert inp.numel() == 0
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


# Explicit default/quick pairs; dtype lists retain device capability filtering.
_SHAPE_DTYPE_CASES = tu.selected_cases(
    [
        ((), torch.float32),
        ((1,), torch.float32),
        ((256,), torch.float32),
        ((1024, 1024), torch.float32),
        ((16, 128, 64, 60), torch.float32),
        ((16, 7, 57, 32, 29), torch.float32),
    ]
    + [((20, 320, 15), dtype) for dtype in _DIM_DTYPES],
    quick=[((2, 19, 7), dtype) for dtype in _DIM_DTYPES],
)


@pytest.mark.dim
@pytest.mark.parametrize("shape, dtype", _SHAPE_DTYPE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges()[:1])
def test_dim_dense_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize(
    "shape",
    tu.selected_cases([(16, 32), (8, 16, 32), (4, 8, 16, 32)], quick=[(2, 19, 7)]),
)
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_noncontiguous_dense(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"]).transpose(0, -1)
    assert not inp.is_contiguous()
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize("case", _COO_CASES)
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_sparse_coo_layouts(case, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, ["-1", "1"])
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)
    # Pure metadata query: the input layout is untouched.
    assert inp.sparse_dim() == len(sparse_shape)
    assert inp.dense_dim() == len(dense_shape)
    assert inp._nnz() == nnz


# Explicit default/quick pairs; dtype lists retain device capability filtering.
_COO_DTYPE_CASES = tu.selected_cases(
    [
        (((3, 4), (), 7), torch.float32),
        (((3, 4), (3,), 8), torch.float32),
    ]
    + [(((12, 9, 3, 6), (4,), 9), dtype) for dtype in _DIM_DTYPES],
    quick=[(((2, 19, 7), (), 8), dtype) for dtype in _DIM_DTYPES],
)


@pytest.mark.dim
@pytest.mark.parametrize("case, dtype", _COO_DTYPE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges()[:1])
def test_dim_sparse_coo_value_ranges(case, value_range, dtype):
    sparse_shape, dense_shape, nnz = case
    inp = _make_coo(sparse_shape, dense_shape, nnz, dtype, value_range)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize("case", _CSR_CASES)
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_sparse_csr_layouts(case, dtype):
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype, ["-1", "1"])
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)
    # Pure metadata query: the input layout is untouched.
    assert inp.dense_dim() == 0
    assert inp.sparse_dim() == 2


# Explicit default/quick pairs; dtype lists retain device capability filtering.
_CSR_DTYPE_CASES = tu.selected_cases(
    [
        (((4, 4), 3), torch.float32),
    ]
    + [(((2, 4, 4), 5), dtype) for dtype in _DIM_DTYPES],
    quick=[(((2, 19, 7), 3), dtype) for dtype in _DIM_DTYPES],
)


@pytest.mark.dim
@pytest.mark.parametrize("case, dtype", _CSR_DTYPE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges()[:1])
def test_dim_sparse_csr_value_ranges(case, value_range, dtype):
    shape, nnz = case
    inp = _make_csr(shape, nnz, dtype, value_range)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_empty_coo(dtype):
    sparse_shape, dense_shape = (3, 4), (5, 6)
    indices = torch.empty(
        len(sparse_shape), 0, dtype=torch.long, device=flag_gems.device
    )
    values = torch.empty(
        (0,) + tuple(dense_shape), dtype=dtype, device=flag_gems.device
    )
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize("shape", [(4, 4), (3, 4, 4)])
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_empty_csr(shape, dtype):
    inp = _make_empty_csr(shape, dtype)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
def test_dim_uncoalesced_coo(dtype):
    sparse_shape, dense_shape = (2, 2), (3,)
    indices = torch.tensor([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]], dtype=torch.long)
    values = _make_input(dtype, (5,) + tuple(dense_shape), ["-1", "1"])
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape + dense_shape, device=flag_gems.device
    )
    assert not inp.is_coalesced()
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DIM_DTYPES))
)
def test_dim_nan_inf_dense(dtype, scenario):
    inp = _repeat(_make_special_input(dtype, scenario), 2)[:6].reshape(2, 3)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DIM_DTYPES))
)
def test_dim_nan_inf_coo(dtype, scenario):
    values = _repeat(_make_special_input(dtype, scenario), 2)[:6]
    indices = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    inp = torch.sparse_coo_tensor(indices, values, (6,), device=flag_gems.device)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DIM_DTYPES))
)
def test_dim_nan_inf_csr(dtype, scenario):
    values = _repeat(_make_special_input(dtype, scenario), 2)[:6]
    crow_indices = torch.tensor([0, 3, 4, 6], dtype=torch.long)
    col_indices = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    inp = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, (3, 4), device=flag_gems.device
    )
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.dim(ref_inp)
    res_out = flag_gems.dim(inp)

    _assert_result(res_out, ref_out)


@pytest.mark.dim
def test_dim_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.dim(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems.dim(3.14)
