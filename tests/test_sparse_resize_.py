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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Resize COO storage in place; non-empty inputs retain their sparse/dense split.
# Empty inputs may change that split.
_RESIZE_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.BOOL_TYPES
)
# (src_shape, sparse_dim, nnz, dst_size, dst_sparse_dim, dst_dense_dim).
_RESIZE_CASES = [
    ((4, 5), 2, 3, [4, 5], 2, 0),
    ((4, 5, 6), 2, 3, [4, 5, 6], 2, 1),
    ((5,), 1, 2, [8], 1, 0),
    ((4, 5), 2, 3, [6, 5], 2, 0),
    ((4, 5), 2, 3, [8, 8], 2, 0),
    ((3, 4, 5), 3, 7, [4, 4, 5], 3, 0),
    ((4, 5, 6), 2, 3, [6, 5, 6], 2, 1),
    ((4, 5), 2, 0, [2, 4, 5], 2, 1),
    ((4, 5), 2, 0, [4, 5, 6], 3, 0),
    ((4, 5), 2, 0, [3, 5], 2, 0),
    ((5,), 1, 0, [2, 3], 2, 0),
]

_VALUE_RANGE_CASES = [
    ((4, 5), 2, 3, [6, 5], 2, 0),
    ((4, 5, 6), 2, 3, [6, 5, 6], 2, 1),
]

_SPARSE_SHAPE_LEVELS = tuple(s for s in tu.selected_shapes() if len(s) > 0)

_NEGATIVE_DTYPES = [torch.float32, torch.int32]

_INVALID_RESIZE_CASES = [
    pytest.param(
        ((4, 5), 2, 3, [4, 5, 6], 2, 0),
        id="invalid_split",
    ),
    pytest.param(
        ((4, 5), 2, 3, [6, 5], 1, 1),
        id="sparse_dim_change_non_empty",
    ),
    pytest.param(
        ((4, 5, 6), 2, 3, [6, 5, 6], 1, 2),
        id="dense_dim_change_non_empty",
    ),
    pytest.param(
        ((4, 5), 2, 3, [3, 5], 2, 0),
        id="shrink_sparse_dim_non_empty",
    ),
    pytest.param(
        ((4, 5, 6), 2, 3, [4, 5, 4], 2, 1),
        id="shrink_dense_dim_non_empty",
    ),
    pytest.param(
        ((4, 5), 2, 3, [4, -5], 2, 0),
        id="negative_size",
    ),
]


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _device(value):
    device = torch.device(value)
    if device.index is None and device.type not in ("cpu", "meta"):
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return device


def _to_target(value, dtype=None, shape=None):
    dtype = value.dtype if dtype is None else dtype
    shape = tuple(value.shape) if shape is None else tuple(shape)
    result = value.to(flag_gems.device)
    if (
        flag_gems.vendor_name == "kunlunxin"
        and dtype in (torch.float64, torch.complex128)
        and result.dtype != dtype
    ):
        pytest.skip(
            "Kunlunxin target transfer narrows requested float64/complex128 storage"
        )
    assert result.dtype == dtype
    assert result.device == _device(flag_gems.device)
    assert tuple(result.shape) == shape
    return result


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and (
        dtype in _FP8_DTYPES or dtype == torch.float64
    ):
        low, high = (tu.resolve_bound(v, dtype) for v in value_range)
        value = (
            torch.full(shape, low, dtype=dtype, device="cpu")
            if low == high
            else torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        )
    else:
        value = tu.make_input(dtype, shape, value_range)
    return _to_target(value, dtype, shape)


def _make_special_input(dtype, scenario):
    # Repeat/slice these recipes on CPU too: sparse storage support does not
    # imply device-side FP8 fill, cast, repeat or slicing kernels exist.
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    return torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu").to(dtype)


def _cpu_values(value, count=None):
    # A uint8 storage view avoids unsupported FP8/complex casts and slicing.
    # For dense growth, copy ONLY the defined original prefix, never its tail.
    assert value.layout == torch.strided
    width = value.element_size()
    storage = value.untyped_storage()
    if count is not None:
        assert 0 <= count <= value.numel()
        if value.is_contiguous():
            raw = (
                torch.empty(0, dtype=torch.uint8, device=value.device)
                .set_(
                    storage, value.storage_offset() * width, (count, width), (width, 1)
                )
                .cpu()
                .clone()
            )
        else:
            chunks = []
            for linear in range(count):
                offset = value.storage_offset()
                for size, stride in reversed(list(zip(value.shape, value.stride()))):
                    offset += (linear % size) * stride
                    linear //= size
                chunks.append(
                    torch.empty(0, dtype=torch.uint8, device=value.device)
                    .set_(storage, offset * width, (width,), (1,))
                    .cpu()
                    .clone()
                )
            raw = (
                torch.stack(chunks)
                if chunks
                else torch.empty((0, width), dtype=torch.uint8)
            )
        result = raw.contiguous().view(value.dtype).reshape(count)
    else:
        raw = (
            torch.empty(0, dtype=torch.uint8, device=value.device)
            .set_(
                storage,
                value.storage_offset() * width,
                (*value.shape, width),
                (*(v * width for v in value.stride()), 1),
            )
            .cpu()
            .clone()
        )
        result = raw.contiguous().view(value.dtype).reshape(value.shape)
    if value.is_conj():
        result = result.conj()
    if value.is_neg():
        result = torch._neg_view(result)
    return result


def _reference(value):
    device = "cpu" if utils.TO_CPU else flag_gems.device
    if value.layout != torch.sparse_coo:
        return _cpu_values(value).to(device)
    result = torch.sparse_coo_tensor(
        _cpu_values(value._indices()).to(device),
        _cpu_values(value._values()).to(device),
        value.shape,
        device=device,
        is_coalesced=value.is_coalesced(),
    )
    assert result.dtype == value.dtype and result.device == _device(device)
    return result


def _assert_values(result, reference, count=None):
    assert result.dtype == reference.dtype
    if count is None:
        assert result.shape == reference.shape
    result = _cpu_values(result, count)
    reference = _cpu_values(reference, count)
    if result.dtype in _FP8_DTYPES:
        result, reference = result.float(), reference.float()
    torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)


def _assert_input(value, shape, sparse_dim, nnz, dtype):
    assert value.layout == torch.sparse_coo and value.dtype == dtype
    assert value.device == _device(flag_gems.device)
    assert tuple(value.shape) == tuple(shape)
    assert value.sparse_dim() == sparse_dim and value._nnz() == nnz
    assert value._indices().dtype == torch.int64
    assert value._indices().device == value.device == value._values().device
    assert value._values().dtype == dtype
    assert tuple(value._values().shape) == (nnz,) + tuple(shape[sparse_dim:])
    return value


def _assert_devices(result, reference):
    assert result.device == _device(flag_gems.device)
    assert reference.device == _device("cpu" if utils.TO_CPU else flag_gems.device)
    for tensor in (result, reference):
        assert tensor._indices().device == tensor.device == tensor._values().device
        assert tensor._indices().dtype == torch.int64
        assert tensor._values().dtype == tensor.dtype


def _candidate_resize(inp, *args):
    try:
        required_device, required_dtype, required_type = (
            inp.device,
            inp.dtype,
            type(inp),
        )
    except Exception as error:
        raise AssertionError("Cannot inspect original input ABI") from error
    try:
        return flag_gems.sparse_resize_(inp, *args)
    finally:
        try:
            assert type(inp) is required_type
            assert inp.device == required_device and inp.dtype == required_dtype
        except Exception as error:
            raise AssertionError("Candidate changed the required input ABI") from error


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0, values=None):
    # Use seeded unique coordinates; supplied values replace the default payload.
    gen = torch.Generator("cpu").manual_seed(seed)
    dense_shape = tuple(shape[sparse_dim:])
    values_shape = (nnz,) + dense_shape
    num_sparse = math.prod(shape[:sparse_dim])
    if nnz == 0:
        indices = torch.empty((sparse_dim, 0), dtype=torch.long)
    else:
        lin = torch.randperm(num_sparse, generator=gen, device="cpu")[:nnz]
        lin = torch.sort(lin).values
        indices = torch.stack(torch.unravel_index(lin, shape[:sparse_dim]), dim=0)
    if values is None:
        # Generate in a wide dtype then cast: randn/randint do not accept every
        # storage dtype (notably fp8), and random_ cannot sample unsigned
        # ranges that include negative values.
        if dtype == torch.bool:
            values = torch.randint(
                0, 2, values_shape, generator=gen, device="cpu"
            ).bool()
        elif dtype.is_floating_point:
            values = torch.randn(values_shape, generator=gen, device="cpu").to(dtype)
        else:
            # Keep the magnitude small so the values stay valid for every
            # integer storage dtype (int8/int16 included).
            values = torch.randint(-5, 6, values_shape, generator=gen, device="cpu").to(
                dtype
            )
    result = torch.sparse_coo_tensor(
        indices.to(flag_gems.device),
        _to_target(values, dtype, values_shape),
        shape,
        device=flag_gems.device,
    )
    return _assert_input(result, shape, sparse_dim, nnz, dtype)


def _assert_resized(res, ref):
    _assert_devices(res, ref)
    assert res.layout == ref.layout
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert res.sparse_dim() == ref.sparse_dim()
    assert res.dense_dim() == ref.dense_dim()
    assert res.is_coalesced() == ref.is_coalesced()
    _assert_values(res._indices(), ref._indices())
    _assert_values(res._values(), ref._values())


_RESIZE_INPUT_DTYPE_CASES = [
    pytest.param(
        dtype,
    )
    for dtype in _RESIZE_DTYPES
]


@pytest.mark.sparse_resize_
@pytest.mark.parametrize(
    "case,dtype",
    [
        pytest.param(
            case,
            dtype,
            id=f"dtype{dtype_index}-case{case_index}",
        )
        for dtype_index, dtype in enumerate(_RESIZE_DTYPES)
        for case_index, case in enumerate(_RESIZE_CASES)
    ],
)
def test_sparse_resize_(case, dtype):
    src_shape, sparse_dim, nnz, size, new_sparse_dim, new_dense_dim = case
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_(
        ref_inp, size, new_sparse_dim, new_dense_dim
    )
    res_out = _candidate_resize(inp, size, new_sparse_dim, new_dense_dim)

    # In-place semantics: the op returns self and mutates self in place.
    assert res_out is inp
    # The mutated input (not only the return value) carries the new structure.
    _assert_resized(res_out, ref_out)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    pytest.param(
        dtype,
        value_range,
    )
    for dtype in _RESIZE_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("case", _VALUE_RANGE_CASES)
@pytest.mark.parametrize(
    "dtype,value_range",
    _DTYPE_RANGE_CASES,
)
def test_sparse_resize_value_ranges(case, dtype, value_range):
    src_shape, sparse_dim, nnz, size, new_sparse_dim, new_dense_dim = case
    dense_shape = tuple(src_shape[sparse_dim:])
    values = _make_input(dtype, (nnz,) + dense_shape, value_range)
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype, values=values)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_(
        ref_inp, size, new_sparse_dim, new_dense_dim
    )
    res_out = _candidate_resize(inp, size, new_sparse_dim, new_dense_dim)

    assert res_out is inp
    _assert_resized(res_out, ref_out)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("shape", _SPARSE_SHAPE_LEVELS)
@pytest.mark.parametrize("dtype", _RESIZE_INPUT_DTYPE_CASES)
def test_sparse_resize_shape_levels(shape, dtype):
    sparse_dim = 1
    dense_dim = len(shape) - 1
    nnz = min(2, shape[0])
    inp = _make_sparse_input(shape, sparse_dim, nnz, dtype)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_(ref_inp, list(shape), sparse_dim, dense_dim)
    res_out = _candidate_resize(inp, list(shape), sparse_dim, dense_dim)

    assert res_out is inp
    _assert_resized(res_out, ref_out)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
        )
        for dtype, scenario in tu.selected_cases(tu.special_value_cases(_RESIZE_DTYPES))
    ],
)
def test_sparse_resize_nan_inf(dtype, scenario):
    src_shape, sparse_dim, nnz = (4, 5, 6), 2, 4
    values = _make_special_input(dtype, scenario).repeat(5)[:24].reshape(nnz, 6)
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype, values=values)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_(ref_inp, [6, 5, 6], 2, 1)
    res_out = _candidate_resize(inp, [6, 5, 6], 2, 1)

    assert res_out is inp
    _assert_resized(res_out, ref_out)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("case", _INVALID_RESIZE_CASES)
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_resize_invalid_raises(case, dtype):
    src_shape, sparse_dim, nnz, size, bad_sparse_dim, bad_dense_dim = case
    inp = _make_sparse_input(src_shape, sparse_dim, nnz, dtype)
    ref_inp = _reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_(ref_inp, size, bad_sparse_dim, bad_dense_dim)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _candidate_resize(inp, size, bad_sparse_dim, bad_dense_dim)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_resize_dense_input_rejected(dtype):
    if dtype.is_floating_point:
        inp = torch.randn((4, 5), dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 5, (4, 5), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_(_reference(inp), [6, 5], 2, 0)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _candidate_resize(inp, [6, 5], 2, 0)


@pytest.mark.sparse_resize_
@pytest.mark.parametrize("dtype", _RESIZE_INPUT_DTYPE_CASES)
def test_sparse_resize_uncoalesced(dtype):
    indices = torch.tensor(
        [[0, 0, 1, 2], [0, 0, 1, 3]], dtype=torch.long, device=flag_gems.device
    )
    values = _make_input(dtype, (4,), ["-1", "1"])
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_(ref_inp, [6, 5], 2, 0)
    res_out = _candidate_resize(inp, [6, 5], 2, 0)

    assert res_out is inp
    _assert_resized(res_out, ref_out)
