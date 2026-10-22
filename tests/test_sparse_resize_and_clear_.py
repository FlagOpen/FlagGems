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

# Resize COO metadata in place and discard all stored entries.
_RESIZE_DTYPES = (
    [torch.float16, torch.float32]
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
)

# (src_shape, src_sparse_dim, src_dense_dim, dst_shape, dst_sparse_dim, dst_dense_dim, nnz).
_RESIZE_CASES = [
    ((4, 5), 2, 0, (4, 5), 2, 0, 5),
    ((4, 5), 2, 0, (4, 5), 1, 1, 5),
    ((2, 3), 2, 0, (4, 5), 2, 0, 4),
    ((5, 5), 2, 0, (2, 3), 2, 0, 7),
    ((4, 5, 6), 2, 1, (4, 5, 6), 2, 1, 5),
    ((4, 5, 6), 1, 2, (3, 6, 7), 1, 2, 4),
    ((4, 5, 6), 2, 1, (4, 5, 6), 1, 2, 5),
    ((5,), 1, 0, (7,), 1, 0, 3),
    ((2, 3, 4, 5), 2, 2, (2, 3, 4, 5), 2, 2, 6),
    ((2, 2, 2, 2, 2), 3, 2, (3, 3, 3, 2, 2), 3, 2, 8),
    ((4, 5), 2, 0, (4, 5), 0, 2, 5),
    ((2, 3, 4), 3, 0, (2, 3, 4), 2, 1, 6),
    ((4, 5), 2, 0, (4, 5), 2, 0, 0),
]

# (dst_shape, sparse_dim, dense_dim) for an empty source.
_EMPTY_SOURCE_TARGETS = [
    ((7,), 1, 0),
    ((2, 3), 2, 0),
    ((4, 5, 6), 2, 1),
    ((4, 5), 0, 2),
    ((3, 3, 3, 3), 3, 1),
]

_NEGATIVE_DTYPES = [torch.float32, torch.int8]

_INVALID_CALLS = [
    pytest.param([4, 5], 1, 0, id="split_too_small"),
    pytest.param([4, 5], 2, 1, id="split_too_large"),
    pytest.param([4, 5], 3, 0, id="split_too_large_sparse"),
    pytest.param([4, 5], -1, 2, id="negative_sparse_dim"),
    pytest.param([4, 5], 2, -1, id="negative_dense_dim"),
    pytest.param([4, -5], 2, 0, id="negative_size"),
]

_SELECTED_RANGES = tu.selected_ranges()

# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_VALUE_RANGE_PAIRS = [
    (dtype, value_range)
    for dtype in _RESIZE_DTYPES
    for value_range in (
        _SELECTED_RANGES[:1] if dtype == torch.bool else _SELECTED_RANGES
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]

_VALUE_RANGE_IDS = [
    f"{str(dtype).replace('torch.', '')}-{'_'.join(value_range)}"
    for dtype, value_range in _VALUE_RANGE_PAIRS
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


def _default_values(dtype, values_shape, gen):
    # Generate on CPU before transferring; FP8 is cast from float32.
    if dtype.is_floating_point:
        base = torch.randn(values_shape, dtype=torch.float32, generator=gen)
        return base.to(dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, values_shape, dtype=dtype, generator=gen)
    if dtype == torch.uint8:
        return torch.randint(0, 6, values_shape, dtype=dtype, generator=gen)
    return torch.randint(-5, 6, values_shape, dtype=dtype, generator=gen)


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
        return flag_gems.sparse_resize_and_clear_(inp, *args)
    finally:
        try:
            assert type(inp) is required_type
            assert inp.device == required_device and inp.dtype == required_dtype
        except Exception as error:
            raise AssertionError("Candidate changed the required input ABI") from error


def _make_sparse_input(shape, sparse_dim, nnz, dtype, seed=0, values=None):
    # Use seeded unique coordinates; supplied values replace the default payload.
    gen = torch.Generator("cpu").manual_seed(seed)
    values_shape = (nnz,) + tuple(shape[sparse_dim:])
    num_sparse = math.prod(shape[:sparse_dim])
    if nnz == 0:
        indices = torch.empty((sparse_dim, 0), dtype=torch.long)
    else:
        lin = torch.randperm(num_sparse, generator=gen, device="cpu")[:nnz]
        lin = torch.sort(lin).values
        indices = torch.stack(torch.unravel_index(lin, shape[:sparse_dim]), dim=0)
    if values is None:
        values = _default_values(dtype, values_shape, gen)
    result = torch.sparse_coo_tensor(
        indices.to(flag_gems.device),
        _to_target(values, dtype, values_shape),
        shape,
        device=flag_gems.device,
    )
    return _assert_input(result, shape, sparse_dim, nnz, dtype)


def _split_for_shape(shape):
    # Use half the target dimensions as sparse dimensions, rounded up.
    ndim = len(shape)
    if ndim == 0:
        return 0, 0
    sparse_dim = (ndim + 1) // 2
    return sparse_dim, ndim - sparse_dim


def _assert_empty_resized(t, shape, sparse_dim, dense_dim, dtype):
    # Check the requested split and empty storage, including its coalesced flag.
    assert t.device == _device(flag_gems.device)
    assert t.layout == torch.sparse_coo
    assert tuple(t.shape) == tuple(shape)
    assert t.dtype == dtype
    assert t.sparse_dim() == sparse_dim
    assert t.dense_dim() == dense_dim
    assert torch.ops.aten._nnz(t) == 0
    assert tuple(torch.ops.aten._indices(t).shape) == (sparse_dim, 0)
    assert tuple(torch.ops.aten._values(t).shape) == (0,) + tuple(shape[sparse_dim:])
    assert t.is_coalesced()


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("case", _RESIZE_CASES)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_(case, dtype):
    src_shape, src_spd, src_dnd, dst_shape, dst_spd, dst_dnd, src_nnz = case
    inp = _make_sparse_input(src_shape, src_spd, src_nnz, dtype)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(
        ref_inp, list(dst_shape), dst_spd, dst_dnd
    )
    res_out = _candidate_resize(inp, list(dst_shape), dst_spd, dst_dnd)

    # In-place semantics: the op returns self and mutates the input in place.
    assert res_out is inp
    # The mutated input (not only the return value) carries the new structure.
    _assert_empty_resized(inp, dst_shape, dst_spd, dst_dnd, dtype)
    _assert_devices(res_out, ref_out)
    _assert_values(res_out._indices(), ref_out._indices())
    _assert_values(res_out._values(), ref_out._values())


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dst_shape,dst_spd,dst_dnd", _EMPTY_SOURCE_TARGETS)
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_empty_source(dst_shape, dst_spd, dst_dnd, dtype):
    inp = _make_sparse_input((4, 5), 2, 0, dtype)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(
        ref_inp, list(dst_shape), dst_spd, dst_dnd
    )
    res_out = _candidate_resize(inp, list(dst_shape), dst_spd, dst_dnd)

    assert res_out is inp
    _assert_empty_resized(inp, dst_shape, dst_spd, dst_dnd, dtype)
    _assert_devices(res_out, ref_out)
    _assert_values(res_out._indices(), ref_out._indices())
    _assert_values(res_out._values(), ref_out._values())


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_uncoalesced(dtype):
    indices = torch.tensor(
        [[0, 0, 1, 2], [0, 0, 1, 3]], dtype=torch.long, device=flag_gems.device
    )
    values = _make_input(dtype, (4,), _SELECTED_RANGES[0]).to(flag_gems.device)
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    assert not inp.is_coalesced()
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(ref_inp, [6, 5], 2, 0)
    res_out = _candidate_resize(inp, [6, 5], 2, 0)

    assert res_out is inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    _assert_devices(res_out, ref_out)
    _assert_values(res_out._indices(), ref_out._indices())
    _assert_values(res_out._values(), ref_out._values())


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("dtype,value_range", _VALUE_RANGE_PAIRS, ids=_VALUE_RANGE_IDS)
def test_sparse_resize_and_clear_value_ranges(dtype, value_range):
    values = _make_input(dtype, (5,), value_range)
    inp = _make_sparse_input((4, 5), 2, 5, dtype, values=values)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(ref_inp, [6, 5], 2, 0)
    res_out = _candidate_resize(inp, [6, 5], 2, 0)

    assert res_out is inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    _assert_devices(res_out, ref_out)
    _assert_values(res_out._indices(), ref_out._indices())
    _assert_values(res_out._values(), ref_out._values())


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_RESIZE_DTYPES))
)
def test_sparse_resize_and_clear_nan_inf(dtype, scenario):
    values = _to_target(_make_special_input(dtype, scenario).repeat(2)[:6], dtype)
    indices = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 4, 4]],
        dtype=torch.long,
        device=flag_gems.device,
    )
    inp = torch.sparse_coo_tensor(indices, values, (4, 5), device=flag_gems.device)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(ref_inp, [6, 5], 2, 0)
    res_out = _candidate_resize(inp, [6, 5], 2, 0)

    assert res_out is inp
    _assert_empty_resized(inp, (6, 5), 2, 0, dtype)
    _assert_devices(res_out, ref_out)
    _assert_values(res_out._indices(), ref_out._indices())
    _assert_values(res_out._values(), ref_out._values())


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _RESIZE_DTYPES)
def test_sparse_resize_and_clear_shape_levels(shape, dtype):
    sparse_dim, dense_dim = _split_for_shape(shape)
    inp = _make_sparse_input((4, 5), 2, 3, dtype)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.sparse_resize_and_clear_(
        ref_inp, list(shape), sparse_dim, dense_dim
    )
    res_out = _candidate_resize(inp, list(shape), sparse_dim, dense_dim)

    assert res_out is inp
    _assert_empty_resized(inp, shape, sparse_dim, dense_dim, dtype)
    _assert_devices(res_out, ref_out)
    _assert_values(res_out._indices(), ref_out._indices())
    _assert_values(res_out._values(), ref_out._values())


@pytest.mark.sparse_resize_and_clear_
@pytest.mark.parametrize("size,sparse_dim,dense_dim", _INVALID_CALLS)
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_sparse_resize_and_clear_invalid_params(size, sparse_dim, dense_dim, dtype):
    inp = _make_sparse_input((4, 5), 2, 3, dtype)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_and_clear_(
            _reference(inp),
            size,
            sparse_dim,
            dense_dim,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _candidate_resize(inp, size, sparse_dim, dense_dim)


@pytest.mark.sparse_resize_and_clear_
def test_sparse_resize_and_clear_non_sparse_input():
    inp = torch.randn((4, 5), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_resize_and_clear_(_reference(inp), [4, 5], 2, 0)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _candidate_resize(inp, [4, 5], 2, 0)
