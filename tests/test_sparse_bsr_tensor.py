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
_VALUE_DTYPES = tu.REQUIRED_DTYPES + [torch.bool, torch.int16, torch.float64]
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_FLOAT_VALUE_DTYPES = [
    dtype
    for dtype in _VALUE_DTYPES
    if dtype.is_floating_point and dtype not in _FP8_DTYPES
]

# (size, block_shape, crow_indices, col_indices).
_BSR_2D_CASES = [
    ((4, 4), (2, 2), [0, 2, 4], [0, 1, 0, 1]),
    ((6, 6), (2, 2), [0, 2, 3, 3], [0, 2, 1]),
    ((4, 6), (2, 3), [0, 1, 2], [0, 1]),
    ((8, 8), (4, 2), [0, 2, 3], [1, 3, 0]),
    ((6, 4), (3, 2), [0, 1, 2], [0, 1]),
    ((10, 12), (2, 4), [0, 3, 5, 6, 6, 6], [0, 1, 2, 0, 2, 1]),
]

# Leading dimensions are batch dimensions; batches share the index grid.
_BSR_BATCHED_CASES = [
    ((2, 6, 6), (2, 3), [0, 2, 4, 4], [0, 1, 0, 1]),
    ((3, 4, 8), (2, 4), [0, 2, 3], [0, 1, 0]),
    ((2, 4, 4), (2, 2), [0, 2, 2], [0, 1]),
    ((2, 8, 12), (4, 3), [0, 1, 3], [0, 1, 3]),
    ((2, 3, 4, 4), (2, 2), [0, 2, 4], [0, 1, 0, 1]),
]

# (size, block_shape, batch); None denotes an unbatched matrix.
_BSR_EMPTY_CASES = [
    ((4, 4), (2, 2), None),
    ((2, 4, 4), (2, 2), 2),
]

# (block_shape, crow_indices, col_indices), with size inferred from indices.
_BSR_2D_INFERRED_CASES = [
    ((2, 2), [0, 2, 4], [0, 1, 0, 1]),
    ((2, 2), [0, 2, 3, 3], [0, 2, 1]),
    ((2, 3), [0, 1, 2], [0, 1]),
]

# Block shape used for the shared shape sweep.
_BLOCK = (2, 2)


def _shape_level_cases():
    # Skip scalars, make 1-D shapes square, and round matrix extents to full blocks.
    cases = []
    for shape in tu.selected_shapes():
        if len(shape) == 0:
            continue
        if len(shape) == 1:
            batch, rows, cols = (), shape[0], shape[0]
        else:
            batch, rows, cols = shape[:-2], shape[-2], shape[-1]
        br, bc = _BLOCK
        rows = max(br, rows // br * br)
        cols = max(bc, cols // bc * bc)
        n_row_blocks = rows // br
        n_col_blocks = cols // bc
        gen = torch.Generator("cpu").manual_seed(len(shape))
        crow = [0]
        col = []
        for r in range(n_row_blocks):
            k = min(1 + (r % 2), n_col_blocks)
            chosen = torch.randperm(n_col_blocks, generator=gen)[:k].sort().values
            col.extend(chosen.tolist())
            crow.append(crow[-1] + k)
        cases.append((batch + (rows, cols), _BLOCK, crow, col))
    return cases


def _storage_bytes(inp):
    storage = inp.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _copy_tensor(inp, device):
    # Transfer physical bytes, then restore the original dense geometry and flags.
    raw = _storage_bytes(inp).to(device=device).clone()
    out = raw.view(inp.dtype).as_strided(inp.shape, inp.stride(), inp.storage_offset())
    if inp.is_conj():
        out = out.conj()
    if inp.is_neg():
        out = torch._neg_view(out)
    out.requires_grad_(inp.requires_grad)
    return out


def _assert_device(inp, expected):
    expected = torch.device(expected)
    assert inp.device.type == expected.type
    if expected.index is not None:
        assert inp.device.index == expected.index


def _to_reference(inp):
    return _copy_tensor(inp, "cpu" if utils.TO_CPU else inp.device)


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
        storage._cdata,
        inp.data_ptr(),
        storage.data_ptr(),
        storage.nbytes(),
    )
    return signature, storage


def _assert_readonly(inp, state, before_bytes):
    signature, storage = _input_state(inp)
    assert signature == state[0]
    assert storage._cdata == state[1]._cdata
    assert torch.equal(_storage_bytes(inp).cpu(), before_bytes)


def _call_candidate(*args, required_device, **kwargs):
    # Hold the original StorageImpl objects and scalar witnesses before the call.
    states = [_input_state(inp) for inp in args[:3]]
    for inp in args[:3]:
        _assert_device(inp, required_device)
    snapshots = [_storage_bytes(inp).cpu().clone() for inp in args[:3]]
    try:
        result = flag_gems.sparse_bsr_tensor(*args, **kwargs)
    finally:
        try:
            for inp, state, before in zip(args[:3], states, snapshots):
                _assert_readonly(inp, state, before)
        except Exception as error:
            raise AssertionError("BSR constructor mutated an input") from error
    assert isinstance(result, torch.Tensor)
    assert result.layout == torch.sparse_bsr
    assert result.dtype == kwargs["dtype"]
    _assert_device(result, required_device)
    if len(args) == 4:
        assert tuple(result.shape) == tuple(args[3])
    return result


def _assert_equal(result, reference, *, required_device, equal_nan=False):
    assert isinstance(result, torch.Tensor) and isinstance(reference, torch.Tensor)
    _assert_device(result, required_device)
    _assert_device(reference, "cpu" if utils.TO_CPU else required_device)
    assert result.dtype == reference.dtype
    assert result.layout == reference.layout
    assert result.shape == reference.shape
    if result.layout == torch.sparse_bsr:
        assert result.sparse_dim() == reference.sparse_dim()
        assert result.dense_dim() == reference.dense_dim()
        assert result._nnz() == reference._nnz()
        for name in ("crow_indices", "col_indices", "values"):
            _assert_equal(
                getattr(result, name)(),
                getattr(reference, name)(),
                required_device=required_device,
                equal_nan=equal_nan,
            )
        return
    assert result.layout == torch.strided
    actual = _copy_tensor(result, "cpu")
    expected = _copy_tensor(reference, "cpu")
    if result.dtype in _FP8_DTYPES:
        actual, expected = actual.float(), expected.float()
    if not (actual.is_floating_point() or actual.is_complex()):
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, equal_nan=equal_nan
        )


def _cpu_encode(dtype):
    return flag_gems.vendor_name in ("ascend", "kunlunxin") and (
        dtype in _FP8_DTYPES or dtype == torch.float64
    )


def _checked_values(inp, dtype, device):
    assert isinstance(inp, torch.Tensor) and inp.layout == torch.strided
    assert inp.dtype == dtype
    _assert_device(inp, device)
    return inp


def _special_values(dtype, scenario, nnz, block):
    if not _cpu_encode(dtype):
        values = tu.make_special_input(dtype, scenario).repeat(4)[:16]
        return _checked_values(values.reshape(nnz, *block), dtype, flag_gems.device)
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    host = torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu").to(dtype)
    host = host.repeat(4)[:16].reshape(nnz, *block)
    return _checked_values(
        _copy_tensor(host, flag_gems.device), dtype, flag_gems.device
    )


def _finish_values(values, dtype, target):
    if _cpu_encode(dtype):
        values = _copy_tensor(values, target)
    return _checked_values(values, dtype, target)


def _make_values(shape, dtype, value_range):
    # Preserve the original clamp rules and testing.make_tensor distribution.
    target = torch.device(flag_gems.device)
    device = "cpu" if _cpu_encode(dtype) else target
    if dtype == torch.bool:
        return _checked_values(
            torch.randint(0, 2, shape, device=target).bool(), dtype, target
        )

    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)

    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        low = max(low, finfo.min)
        high = min(high, finfo.max)
        if not low < high:
            return _finish_values(
                torch.full(shape, low, dtype=dtype, device=device), dtype, target
            )
    else:
        low, high = int(low), int(high)
        dmin, dmax = tu.dtype_bounds(dtype)
        low = max(low, int(dmin))
        high = min(max(high, low), int(dmax))
        if low >= high:
            return _finish_values(
                torch.full(shape, low, dtype=dtype, device=device), dtype, target
            )

    values = torch.testing.make_tensor(
        shape, dtype=dtype, device=device, low=low, high=high
    )
    return _finish_values(values, dtype, target)


def _make_bsr_values(nnz, block, dtype, batch=None, value_range=("-1", "1")):
    if batch is None:
        shape = (nnz, block[0], block[1])
    elif isinstance(batch, tuple):
        shape = batch + (nnz, block[0], block[1])
    else:
        shape = (batch,) + (nnz, block[0], block[1])
    return _make_values(shape, dtype, value_range)


def _assert_bsr_structure(out, size, block, nnz, dtype, batch=None):
    assert out.layout == torch.sparse_bsr
    assert tuple(out.shape) == tuple(size)
    assert out.dtype == dtype
    assert out.sparse_dim() == 2
    assert out.dense_dim() == 0
    assert out._nnz() == nnz
    if batch is None:
        assert tuple(out.values().shape) == (nnz, block[0], block[1])
    elif isinstance(batch, tuple):
        assert tuple(out.values().shape) == batch + (nnz, block[0], block[1])
    else:
        assert tuple(out.values().shape) == (batch,) + (nnz, block[0], block[1])
    n_row_blocks = size[-2] // block[0]
    n_col_blocks = size[-1] // block[1]
    assert out.crow_indices().shape == size[:-2] + (n_row_blocks + 1,)
    assert (out.col_indices() < n_col_blocks).all()
    assert (out.col_indices() >= 0).all()


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_2D_CASES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value_size(case, dtype, value_range):
    size, block, crow, col = case
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, dtype, value_range=value_range)
    expected_device = values.device
    ref_crow = _to_reference(crow_t)
    ref_col = _to_reference(col_t)
    ref_values = _to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=crow_t.device,
        required_device=expected_device,
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    _assert_equal(res_out, ref_out, required_device=expected_device)
    _assert_equal(
        res_out.crow_indices(), ref_out.crow_indices(), required_device=expected_device
    )
    _assert_equal(
        res_out.col_indices(), ref_out.col_indices(), required_device=expected_device
    )
    # The constructor reads its inputs; it must not mutate them.
    _assert_equal(crow_t, ref_crow, required_device=expected_device)
    _assert_equal(col_t, ref_col, required_device=expected_device)
    _assert_equal(values, ref_values, required_device=expected_device)


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_BATCHED_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value_size_batched(case, dtype, value_range):
    size, block, crow, col = case
    batch = size[:-2]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    crow_t = crow_t.expand(batch + (len(crow),)).contiguous()
    col_t = col_t.expand(batch + (nnz,)).contiguous()
    values = _make_bsr_values(nnz, block, dtype, batch=batch, value_range=value_range)
    expected_device = values.device
    ref_crow = _to_reference(crow_t)
    ref_col = _to_reference(col_t)
    ref_values = _to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=crow_t.device,
        required_device=expected_device,
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype, batch=batch)
    _assert_equal(res_out, ref_out, required_device=expected_device)
    _assert_equal(
        res_out.crow_indices(), ref_out.crow_indices(), required_device=expected_device
    )
    _assert_equal(
        res_out.col_indices(), ref_out.col_indices(), required_device=expected_device
    )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_EMPTY_CASES)
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
def test_sparse_bsr_tensor_crow_col_value_size_empty(case, dtype):
    size, block, batch = case
    n_row_blocks = size[-2] // block[0]
    crow_t = torch.zeros(
        size[:-2] + (n_row_blocks + 1,), dtype=torch.long, device=flag_gems.device
    )
    col_t = torch.empty(size[:-2] + (0,), dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(0, block, dtype, batch=batch)
    expected_device = values.device
    ref_crow = _to_reference(crow_t)
    ref_col = _to_reference(col_t)
    ref_values = _to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=crow_t.device,
        required_device=expected_device,
    )

    _assert_bsr_structure(res_out, size, block, 0, dtype, batch=batch)
    _assert_equal(res_out, ref_out, required_device=expected_device)
    _assert_equal(
        res_out.crow_indices(), ref_out.crow_indices(), required_device=expected_device
    )
    _assert_equal(
        res_out.col_indices(), ref_out.col_indices(), required_device=expected_device
    )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _BSR_2D_INFERRED_CASES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_crow_col_value(case, dtype, value_range):
    block, crow, col = case
    nnz = len(col)
    size = ((len(crow) - 1) * block[0], (max(col) + 1) * block[1])
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, dtype, value_range=value_range)
    expected_device = values.device
    ref_crow = _to_reference(crow_t)
    ref_col = _to_reference(col_t)
    ref_values = _to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        dtype=dtype,
        device=crow_t.device,
        required_device=expected_device,
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    _assert_equal(res_out, ref_out, required_device=expected_device)
    _assert_equal(
        res_out.crow_indices(), ref_out.crow_indices(), required_device=expected_device
    )
    _assert_equal(
        res_out.col_indices(), ref_out.col_indices(), required_device=expected_device
    )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("case", _shape_level_cases())
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_sparse_bsr_tensor_shape_levels(case, dtype, value_range):
    size, block, crow, col = case
    batch = size[:-2]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    crow_t = crow_t.expand(batch + (len(crow),)).contiguous()
    col_t = col_t.expand(batch + (nnz,)).contiguous()
    values = _make_bsr_values(nnz, block, dtype, batch=batch, value_range=value_range)
    expected_device = values.device
    ref_crow = _to_reference(crow_t)
    ref_col = _to_reference(col_t)
    ref_values = _to_reference(values)

    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow, ref_col, ref_values, list(size), dtype=dtype, device=ref_crow.device
    )
    res_out = _call_candidate(
        crow_t,
        col_t,
        values,
        list(size),
        dtype=dtype,
        device=crow_t.device,
        required_device=expected_device,
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype, batch=batch)
    _assert_equal(res_out, ref_out, required_device=expected_device)
    _assert_equal(
        res_out.crow_indices(), ref_out.crow_indices(), required_device=expected_device
    )
    _assert_equal(
        res_out.col_indices(), ref_out.col_indices(), required_device=expected_device
    )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_VALUE_DTYPES))
)
def test_sparse_bsr_tensor_nan_inf_values(dtype, scenario):
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    values = _special_values(dtype, scenario, nnz, block)
    expected_device = values.device
    ref_values = _to_reference(values)

    ref_crow = torch.tensor([0, 2, 4], dtype=torch.long, device=ref_values.device)
    ref_col = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=ref_values.device)
    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow,
        ref_col,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_values.device,
    )
    res_out = _call_candidate(
        torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device),
        torch.tensor([0, 1, 0, 1], dtype=torch.long, device=flag_gems.device),
        values,
        list(size),
        dtype=dtype,
        device=values.device,
        required_device=expected_device,
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    _assert_equal(
        res_out.crow_indices(),
        ref_out.crow_indices(),
        required_device=expected_device,
        equal_nan=True,
    )
    _assert_equal(
        res_out.col_indices(),
        ref_out.col_indices(),
        required_device=expected_device,
        equal_nan=True,
    )
    _assert_equal(
        res_out.values(),
        ref_out.values(),
        required_device=expected_device,
        equal_nan=True,
    )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("dtype", _FLOAT_VALUE_DTYPES)
def test_sparse_bsr_tensor_boundary_values(dtype):
    # Include exact finfo endpoints that random range sampling may miss.
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    finfo = torch.finfo(dtype)
    specials = torch.tensor(
        [finfo.min, finfo.max, 0.0, -0.0, 1.0, -1.0],
        dtype=dtype,
        device="cpu" if _cpu_encode(dtype) else flag_gems.device,
    )
    n_elems = nnz * block[0] * block[1]
    values = specials.repeat((n_elems + specials.numel() - 1) // specials.numel())[
        :n_elems
    ]
    values = values.reshape(nnz, block[0], block[1])
    values = _finish_values(values, dtype, torch.device(flag_gems.device))
    expected_device = values.device
    ref_values = _to_reference(values)

    ref_crow = torch.tensor([0, 2, 4], dtype=torch.long, device=ref_values.device)
    ref_col = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=ref_values.device)
    ref_out = torch.ops.aten.sparse_bsr_tensor(
        ref_crow,
        ref_col,
        ref_values,
        list(size),
        dtype=dtype,
        device=ref_values.device,
    )
    res_out = _call_candidate(
        torch.tensor([0, 2, 4], dtype=torch.long, device=flag_gems.device),
        torch.tensor([0, 1, 0, 1], dtype=torch.long, device=flag_gems.device),
        values,
        list(size),
        dtype=dtype,
        device=values.device,
        required_device=expected_device,
    )

    _assert_bsr_structure(res_out, size, block, nnz, dtype)
    _assert_equal(res_out, ref_out, required_device=expected_device)


@pytest.mark.sparse_bsr_tensor
def test_sparse_bsr_tensor_rejects_dtype_mismatch():
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, torch.float16)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            ref_crow,
            ref_col,
            ref_values,
            list(size),
            dtype=torch.float32,
            device=ref_crow.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_bsr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32, device=crow_t.device
        )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32])
def test_sparse_bsr_tensor_rejects_missing_dtype(dtype):
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, dtype)
    ref_crow = tu.to_reference(crow_t)
    ref_col = tu.to_reference(col_t)
    ref_values = tu.to_reference(values)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            ref_crow, ref_col, ref_values, list(size), device=ref_crow.device
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_bsr_tensor(
            crow_t, col_t, values, list(size), device=crow_t.device
        )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="cross-device construction requires a non-CPU device",
)
def test_sparse_bsr_tensor_rejects_device_mismatch():
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, torch.float32)
    cpu_values = values.cpu()

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            crow_t,
            col_t,
            cpu_values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_bsr_tensor(
            crow_t,
            col_t,
            cpu_values,
            list(size),
            dtype=torch.float32,
            device=crow_t.device,
        )


@pytest.mark.sparse_bsr_tensor
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="device inference from component tensors only applies to accelerators",
)
def test_sparse_bsr_tensor_rejects_missing_device():
    size, block, crow, col = _BSR_2D_CASES[0]
    nnz = len(col)
    crow_t = torch.tensor(crow, dtype=torch.long, device=flag_gems.device)
    col_t = torch.tensor(col, dtype=torch.long, device=flag_gems.device)
    values = _make_bsr_values(nnz, block, torch.float32)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_bsr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32
        )
    with pytest.raises((TypeError, ValueError, NotImplementedError, RuntimeError)):
        flag_gems.sparse_bsr_tensor(
            crow_t, col_t, values, list(size), dtype=torch.float32
        )
