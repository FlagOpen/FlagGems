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
#
# Correctness tests for aten::_sparse_coo_tensor_with_dims.
#
# The operator only allocates an empty COO tensor: it has no tensor operand and
# stores no elements (nnz == 0). The spec's value-range, nan/inf, broadcast and
# backward dimensions therefore have nothing to attach to -- there is no input
# value to range over, no stored element that could be special, no second
# operand to broadcast against, and the result is a leaf allocation
# (requires_grad False, grad_fn None). The coverage below is the allocation
# contract instead: rank and sparse_dim/dense_dim split, dtype, device,
# indices/values metadata, the .out overload, the schema defaults and the
# invalid-call negatives.

import pytest
import torch

import flag_gems

from . import conftest as cfg
from . import test_utils as tu

_SPARSE_COO = torch.sparse_coo

# The nine spec dtypes plus float64/bool. Every entry was probed native-valid
# for this allocator with layout=torch.sparse_coo on the active backend, so no
# supported dtype is filtered out.
GRID_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
    torch.bool,
]


def _expected_device(device=None):
    device = torch.device(flag_gems.device if device is None else device)
    if device.type not in ("cpu", "meta") and device.index is None:
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return device


def _candidate(*args, required_device=None, **kwargs):
    out = kwargs.get("out")
    if required_device is None:
        # Non-tensor out negatives must reach the candidate itself.
        device = out.device if isinstance(out, torch.Tensor) else kwargs.get("device")
        required_device = _expected_device(device)
    result = flag_gems._sparse_coo_tensor_with_dims(*args, **kwargs)
    try:
        assert isinstance(result, torch.Tensor)
        assert result.layout == _SPARSE_COO
        assert result.device == required_device
        assert result._indices().device == required_device
        assert result._values().device == required_device
    except Exception as error:
        # A malformed returned object is not the native rejection expected by
        # the negative cases. Keep the actual candidate call outside this catch.
        raise AssertionError("candidate returned an invalid COO allocation") from error
    return result


def _cpu_component(tensor, *, required_device, dtype):
    assert tensor.device == required_device
    assert tensor.dtype == dtype and tensor.layout == torch.strided
    result = torch.empty(0, dtype=dtype, device="cpu").set_(
        tensor.untyped_storage().cpu(),
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
    )
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        result = result.to(torch.float32)
    if tensor.is_conj():
        result = result.conj()
    if tensor.is_neg():
        result = torch._neg_view(result)
    return result


def _dense_observation(tensor, *, required_device, dtype):
    assert tensor.device == required_device
    assert tensor.dtype == dtype and tensor.layout == _SPARSE_COO
    indices = _cpu_component(
        tensor._indices(), required_device=required_device, dtype=torch.int64
    )
    values = _cpu_component(
        tensor._values(), required_device=required_device, dtype=dtype
    )
    # Observe the candidate's actual empty components. FP8 is decoded only
    # after checking its original ABI, avoiding auxiliary FP8 zero-fill kernels.
    observed = torch.sparse_coo_tensor(
        indices,
        values,
        size=tuple(tensor.shape),
        dtype=values.dtype,
        device="cpu",
        is_coalesced=tensor.is_coalesced(),
    )
    assert observed.sparse_dim() == tensor.sparse_dim()
    assert observed.dense_dim() == tensor.dense_dim()
    assert observed._nnz() == tensor._nnz()
    return observed.to_dense()


def _filled_values(shape, dtype, device):
    device = _expected_device(device)
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        values = torch.ones(shape, dtype=dtype, device=device)
    else:
        encoded = torch.ones(shape, dtype=torch.float32, device="cpu").to(dtype)
        payload = torch.empty(0, dtype=torch.uint8, device="cpu").set_(
            encoded.untyped_storage(), 0, (encoded.untyped_storage().nbytes(),), (1,)
        )
        transferred = payload.to(device)
        values = torch.empty(0, dtype=dtype, device=device).set_(
            transferred.untyped_storage(),
            encoded.storage_offset(),
            encoded.size(),
            encoded.stride(),
        )
    assert values.dtype == dtype and values.device == device
    assert tuple(values.shape) == tuple(shape)
    return values


def _reference_device():
    # Follow the session's configured reference device; the candidate always
    # allocates on flag_gems.device.
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _call_native(sparse_dim, dense_dim, size, dtype, **kwargs):
    return torch.ops.aten._sparse_coo_tensor_with_dims(
        sparse_dim,
        dense_dim,
        size,
        dtype=dtype,
        layout=_SPARSE_COO,
        device=_reference_device(),
        **kwargs,
    )


def _call_candidate(sparse_dim, dense_dim, size, dtype, *, required_device, **kwargs):
    return _candidate(
        sparse_dim,
        dense_dim,
        size,
        dtype=dtype,
        layout=_SPARSE_COO,
        device=flag_gems.device,
        required_device=required_device,
        **kwargs,
    )


def _assert_empty_coo(
    res_out, ref_out, sparse_dim, dense_dim, size, dtype, *, required_device
):
    # Metadata of one candidate/native pair. A zero-nnz COO tensor stores no
    # element values, so indices/values are its only content and are compared
    # with the shared exact assertion.
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.layout == ref_out.layout == _SPARSE_COO
    assert tuple(res_out.shape) == tuple(ref_out.shape) == tuple(size)
    assert res_out.sparse_dim() == ref_out.sparse_dim() == sparse_dim
    assert res_out.dense_dim() == ref_out.dense_dim() == dense_dim
    assert res_out._nnz() == ref_out._nnz() == 0
    assert res_out.is_coalesced() and ref_out.is_coalesced()
    assert not res_out._is_view() and not ref_out._is_view()
    assert res_out.requires_grad is False and res_out.grad_fn is None
    assert res_out.device == required_device
    reference_device = torch.device("cpu") if cfg.TO_CPU else required_device
    assert ref_out.device == reference_device
    res_indices, ref_indices = res_out._indices(), ref_out._indices()
    res_values, ref_values = res_out._values(), ref_out._values()
    assert tuple(res_indices.shape) == tuple(ref_indices.shape) == (sparse_dim, 0)
    assert res_indices.dtype == ref_indices.dtype == torch.int64
    assert (
        tuple(res_values.shape)
        == tuple(ref_values.shape)
        == (0,) + tuple(size[sparse_dim:])
    )
    assert res_values.dtype == ref_values.dtype == dtype
    tu.assert_result_equal(
        _cpu_component(res_indices, required_device=required_device, dtype=torch.int64),
        _cpu_component(
            ref_indices, required_device=reference_device, dtype=torch.int64
        ),
    )
    tu.assert_result_equal(
        _cpu_component(res_values, required_device=required_device, dtype=dtype),
        _cpu_component(ref_values, required_device=reference_device, dtype=dtype),
    )


@pytest.mark.underscore_sparse_coo_tensor_with_dims
@pytest.mark.parametrize("dtype", GRID_DTYPES)
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_sparse_coo_tensor_with_dims(shape, dtype):
    required_device = _expected_device()
    # dense_dim = 0 is one valid split per spec rank; the dense-tail and
    # dense-only splits live in test_sparse_coo_tensor_with_dims_dim_split.
    sparse_dim, dense_dim = len(shape), 0
    size = list(shape)
    ref_out = _call_native(sparse_dim, dense_dim, size, dtype)
    res_out = _call_candidate(
        sparse_dim, dense_dim, size, dtype, required_device=required_device
    )
    _assert_empty_coo(
        res_out,
        ref_out,
        sparse_dim,
        dense_dim,
        size,
        dtype,
        required_device=required_device,
    )


# Splits with a dense tail (dense_dim > 0) and the dense-only split
# (sparse_dim == 0); dense_dim == 0 for the spec shapes is the main grid. Every
# row satisfies sparse_dim + dense_dim == len(size) and is one workload.
SPLIT_CASES = tu.selected_cases(
    [
        ((2, 19, 7), (2, 1)),
        ((2, 19, 7), (1, 2)),
        ((2, 19, 7), (0, 3)),
        ((1024, 1024), (1, 1)),
        ((1024, 1024), (0, 2)),
        ((20, 320, 15), (2, 1)),
        ((20, 320, 15), (1, 2)),
        ((20, 320, 15), (0, 3)),
        ((16, 128, 64, 60), (3, 1)),
        ((16, 128, 64, 60), (2, 2)),
        ((16, 128, 64, 60), (0, 4)),
        ((16, 7, 57, 32, 29), (3, 2)),
        ((16, 7, 57, 32, 29), (0, 5)),
    ],
    quick=[
        ((2, 19, 7), (2, 1)),
        ((2, 19, 7), (1, 2)),
        ((2, 19, 7), (0, 3)),
    ],
)


@pytest.mark.underscore_sparse_coo_tensor_with_dims
@pytest.mark.parametrize("dtype", GRID_DTYPES)
@pytest.mark.parametrize("shape,split", SPLIT_CASES)
def test_sparse_coo_tensor_with_dims_dim_split(shape, split, dtype):
    required_device = _expected_device()
    sparse_dim, dense_dim = split
    size = list(shape)
    ref_out = _call_native(sparse_dim, dense_dim, size, dtype)
    res_out = _call_candidate(
        sparse_dim, dense_dim, size, dtype, required_device=required_device
    )
    _assert_empty_coo(
        res_out,
        ref_out,
        sparse_dim,
        dense_dim,
        size,
        dtype,
        required_device=required_device,
    )


# Allocation-free rows for the remaining valid size branches: rank-0 size, a
# singleton extent, zero extents at either end, a zero-size dense tail and a
# dense-only split. They store nothing per element, so both execution levels
# keep all of them.
SMALL_CASES = [
    (0, 0, []),
    (1, 0, [1]),
    (2, 0, [0, 4]),
    (2, 0, [3, 0]),
    (1, 1, [3, 0]),
    (0, 2, [0, 5]),
]


@pytest.mark.underscore_sparse_coo_tensor_with_dims
@pytest.mark.parametrize("dtype", GRID_DTYPES)
@pytest.mark.parametrize("sparse_dim,dense_dim,size", SMALL_CASES)
def test_sparse_coo_tensor_with_dims_small_sizes(sparse_dim, dense_dim, size, dtype):
    required_device = _expected_device()
    ref_out = _call_native(sparse_dim, dense_dim, size, dtype)
    res_out = _call_candidate(
        sparse_dim, dense_dim, size, dtype, required_device=required_device
    )
    _assert_empty_coo(
        res_out,
        ref_out,
        sparse_dim,
        dense_dim,
        size,
        dtype,
        required_device=required_device,
    )


# to_dense() materializes the whole size, so this value-level check uses small
# shapes; the spec's larger shapes keep their metadata coverage above.
DENSE_SHAPES = tu.selected_cases(
    [(1,), (4, 5), (2, 19, 7)],
    quick=[(2, 19, 7)],
)


@pytest.mark.underscore_sparse_coo_tensor_with_dims
@pytest.mark.parametrize("dtype", GRID_DTYPES)
@pytest.mark.parametrize("shape", DENSE_SHAPES)
def test_sparse_coo_tensor_with_dims_dense_observation(shape, dtype):
    required_device = _expected_device()
    sparse_dim, dense_dim = len(shape), 0
    size = list(shape)
    ref_out = _call_native(sparse_dim, dense_dim, size, dtype)
    res_out = _call_candidate(
        sparse_dim, dense_dim, size, dtype, required_device=required_device
    )
    _assert_empty_coo(
        res_out,
        ref_out,
        sparse_dim,
        dense_dim,
        size,
        dtype,
        required_device=required_device,
    )
    # The allocation is all zeros, not uninitialized memory.
    tu.assert_result_equal(
        _dense_observation(res_out, required_device=required_device, dtype=dtype),
        _dense_observation(
            ref_out,
            required_device=torch.device("cpu") if cfg.TO_CPU else required_device,
            dtype=dtype,
        ),
    )


def _out_buffer(state, dtype, device, sparse_dim, size):
    device = _expected_device(device)
    # Values are 1-D (nnz) because dense_dim == 0 here; only explicitly written
    # elements are used, so nothing uninitialized is compared. The filled buffer
    # is lexicographically sorted with no duplicate column and is flagged
    # coalesced directly: sparse coalesce has no FP8 kernel on this backend, and
    # the pre-call state is only there to be reset.
    tail = tuple(size[sparse_dim:])
    if state == "filled":
        indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64, device=device)
        values = _filled_values((2,) + tail, dtype, device)
        result = torch.sparse_coo_tensor(
            indices, values, size=tuple(size), device=device, is_coalesced=True
        )
    else:
        indices = torch.empty((sparse_dim, 0), dtype=torch.int64, device=device)
        values = torch.empty((0,) + tail, dtype=dtype, device=device)
        result = torch.sparse_coo_tensor(
            indices, values, size=tuple(size), device=device
        )
    assert result.dtype == dtype and result.device == device
    assert result.layout == _SPARSE_COO and tuple(result.shape) == tuple(size)
    assert result._indices().device == result._values().device == device
    return result


# The .out overload writes into the caller's buffer and returns that same
# object; the "filled" buffer starts coalesced with nnz > 0 so the reset of the
# stored content is observable. Both states are allocation-free, so quick keeps
# them. The .out schema has no dtype argument: the buffer keeps its own dtype.
OUT_STATES = ["empty", "filled"]


@pytest.mark.underscore_sparse_coo_tensor_with_dims
@pytest.mark.parametrize("dtype", GRID_DTYPES)
@pytest.mark.parametrize("state", OUT_STATES)
def test_sparse_coo_tensor_with_dims_out(state, dtype):
    required_device = _expected_device()
    sparse_dim, dense_dim, size = 2, 0, [3, 4]
    ref_buf = _out_buffer(state, dtype, _reference_device(), sparse_dim, size)
    res_buf = _out_buffer(state, dtype, flag_gems.device, sparse_dim, size)
    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims.out(
        sparse_dim, dense_dim, size, out=ref_buf
    )
    res_out = _candidate(
        sparse_dim, dense_dim, size, out=res_buf, required_device=required_device
    )
    assert res_out is res_buf
    _assert_empty_coo(
        res_out,
        ref_out,
        sparse_dim,
        dense_dim,
        size,
        dtype,
        required_device=required_device,
    )


# Bool parameter: both values are allocation-free, so quick keeps the pair.
PIN_MEMORY_DTYPES = [torch.float32, torch.float16, torch.int32]
PIN_MEMORY_VALUES = [False, True]


@pytest.mark.underscore_sparse_coo_tensor_with_dims
@pytest.mark.parametrize("dtype", PIN_MEMORY_DTYPES)
@pytest.mark.parametrize("pin_memory", PIN_MEMORY_VALUES)
def test_sparse_coo_tensor_with_dims_pin_memory(pin_memory, dtype):
    required_device = _expected_device()
    sparse_dim, dense_dim, size = 2, 0, [3, 4]
    ref_out = _call_native(sparse_dim, dense_dim, size, dtype, pin_memory=pin_memory)
    res_out = _call_candidate(
        sparse_dim,
        dense_dim,
        size,
        dtype,
        required_device=required_device,
        pin_memory=pin_memory,
    )
    _assert_empty_coo(
        res_out,
        ref_out,
        sparse_dim,
        dense_dim,
        size,
        dtype,
        required_device=required_device,
    )


@pytest.mark.underscore_sparse_coo_tensor_with_dims
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_sparse_coo_tensor_with_dims_schema_default(shape):
    required_device = _expected_device()
    # dtype is omitted so the schema default (float32) is exercised. layout is
    # always given because the native layout default (None) has no sparse kernel
    # and raises NotImplementedError, and device is explicit because the native
    # default is CPU while the candidate allocates on flag_gems.device.
    sparse_dim, dense_dim, size = len(shape), 0, list(shape)
    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims(
        sparse_dim, dense_dim, size, layout=_SPARSE_COO, device=_reference_device()
    )
    res_out = _candidate(
        sparse_dim,
        dense_dim,
        size,
        layout=_SPARSE_COO,
        device=flag_gems.device,
        required_device=required_device,
    )
    _assert_empty_coo(
        res_out,
        ref_out,
        sparse_dim,
        dense_dim,
        size,
        torch.float32,
        required_device=required_device,
    )


# Invalid-call rows; every row is kept in both execution levels and asserts only
# the candidate's own exception.
_NEGATIVE_ROWS = [
    ("rank_mismatch", (2, 0, [3, 4, 5]), {}, None),
    ("negative_extent", (1, 0, [-3]), {}, None),
    ("scalar_size", (2, 0, 3), {}, None),
    ("non_int_sparse_dim", (1.5, 0, [3]), {}, None),
    (
        "dense_layout",
        (2, 0, [3, 4]),
        {"dtype": torch.float32, "device": flag_gems.device, "layout": torch.strided},
        None,
    ),
    ("out_dense_buffer", (2, 0, [3, 4]), {}, "dense"),
    ("out_non_tensor", (2, 0, [3, 4]), {}, "non_tensor"),
    ("out_size_mismatch", (2, 0, [3, 4]), {}, "size_mismatch"),
]


@pytest.mark.underscore_sparse_coo_tensor_with_dims
@pytest.mark.parametrize(
    "args,kwargs,out_kind",
    [(row[1], row[2], row[3]) for row in _NEGATIVE_ROWS],
    ids=[row[0] for row in _NEGATIVE_ROWS],
)
def test_sparse_coo_tensor_with_dims_invalid(args, kwargs, out_kind):
    call_kwargs = (
        {"dtype": torch.float32, "layout": _SPARSE_COO, "device": flag_gems.device}
        if out_kind is None
        else {}
    )
    call_kwargs.update(kwargs)
    if out_kind == "dense":
        call_kwargs["out"] = torch.zeros(3, 4, device=flag_gems.device)
    elif out_kind == "non_tensor":
        call_kwargs["out"] = "not-a-tensor"
    elif out_kind == "size_mismatch":
        call_kwargs["out"] = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.int64, device=flag_gems.device),
            torch.empty((0,), dtype=torch.float32, device=flag_gems.device),
            size=(9, 9),
            device=flag_gems.device,
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _candidate(*args, **call_kwargs)
