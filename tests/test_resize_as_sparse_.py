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

# aten::resize_as_sparse_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)
# resizes sparse COO ``self`` in place to the template's shape and sparse/dense
# split and returns ``self``; the template contributes sizes only. Native-probed
# exemptions:
# * backward -- the operator registers no derivative ("derivative for
#   aten::resize_as_sparse_ is not implemented").
# * broadcast -- sparse COO has no broadcasting; the template must keep the
#   rank and split counts for non-empty ``self``; its extents may grow.
# * parameter sweeps -- the schema takes two tensors and has no bool/int/float
#   parameter with a default or an interesting boundary.
# A native probe of this overload accepted every dtype listed here, so there is
# no unsupported-dtype negative case to add.
_DTYPES = [
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
] + utils.BOOL_TYPES

_DEFAULT_RANGE = ["-1", "1"]

# Columns: self_shape, self_sparse_dim, nnz, coalesced, template_kind,
# template_shape, template_sparse_dim, retains_storage, grows_dense.
_ALL_CASES = [
    pytest.param(
        ((4, 5), 2, 3, True, "sparse", (4, 5), 2, True, False), id="same_size"
    ),
    pytest.param(
        ((4, 5), 2, 3, True, "sparse", (6, 5), 2, True, False), id="grow_sparse"
    ),
    pytest.param(
        ((4, 5), 2, 4, False, "sparse", (6, 5), 2, False, False), id="uncoalesced"
    ),
    pytest.param(
        ((4, 5, 6), 2, 3, True, "sparse", (6, 5, 6), 2, False, False),
        id="grow_sparse_with_dense",
    ),
    pytest.param(
        ((4, 5, 6), 2, 3, True, "sparse", (4, 5, 9), 2, False, True),
        id="grow_dense",
    ),
    pytest.param(
        ((4, 5), 2, 0, True, "dense", (6, 7), 0, False, False), id="empty_to_dense"
    ),
    pytest.param(((), 0, 0, True, "sparse", (), 0, False, False), id="rank0_empty"),
    pytest.param(
        ((3, 4, 5), 3, 7, True, "sparse", (4, 4, 5), 3, False, False),
        id="three_sparse_dims",
    ),
    pytest.param(
        ((2, 3, 4, 5), 2, 4, True, "sparse", (2, 5, 4, 5), 2, False, False),
        id="two_dense_dims",
    ),
]

# Columns: self_shape, self_sparse_dim, nnz, template_kind, template_shape,
# template_sparse_dim. Every row is a valid native call for its ``self`` that
# the reference rejects because the template would shrink a non-empty tensor or
# change the sparse/dense split; the candidate must reject it too.
_NEGATIVE_CASES = [
    pytest.param(((4, 5), 2, 3, "sparse", (3, 5), 2), id="shrink_sparse"),
    pytest.param(((4, 5, 6), 2, 3, "sparse", (4, 5, 4), 2), id="shrink_dense"),
    pytest.param(((4, 5), 2, 3, "sparse", (6, 5), 1), id="sparse_dim_count"),
    pytest.param(((4, 5, 6), 2, 3, "sparse", (6, 6), 2), id="dense_dim_count"),
    pytest.param(((4, 5), 2, 3, "dense", (6, 7), 0), id="dense_template"),
]

_NEGATIVE_DTYPES = [torch.float32, torch.int32]


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


def _require_native_resize(inp, template):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and inp.dtype == torch.int64
        and inp._nnz() > 0
        and tuple(inp.shape[inp.sparse_dim() :])
        != tuple(template.shape[inp.sparse_dim() :])
    ):
        pytest.skip(
            "Kunlunxin native int64 dense growth corrupts independent live GPU storage"
        )


def _template_state(value):
    parts = (
        (value._indices(), value._values())
        if value.layout == torch.sparse_coo
        else (value,)
    )
    metadata = (
        type(value),
        value.dtype,
        value.device,
        value.layout,
        tuple(value.shape),
        value.requires_grad,
        value._version,
    )
    if value.layout == torch.sparse_coo:
        metadata += (value.sparse_dim(), value.dense_dim(), value.is_coalesced())
    return metadata, [
        (
            part.untyped_storage(),
            (
                type(part),
                part.dtype,
                part.device,
                tuple(part.shape),
                tuple(part.stride()),
                part.storage_offset(),
                part.is_conj(),
                part.is_neg(),
                part.requires_grad,
                part._version,
                part.untyped_storage()._cdata,
                part.data_ptr(),
                part.untyped_storage().data_ptr(),
                part.untyped_storage().nbytes(),
            ),
            _cpu_values(part),
        )
        for part in parts
    ]


def _candidate_resize(inp, template):
    try:
        required_device, required_dtype, required_type = (
            inp.device,
            inp.dtype,
            type(inp),
        )
        before_meta, before_parts = _template_state(template)
    except Exception as error:
        raise AssertionError("Cannot inspect original template") from error
    try:
        return flag_gems.resize_as_sparse_(inp, template)
    finally:
        try:
            assert type(inp) is required_type
            assert inp.device == required_device and inp.dtype == required_dtype
            after_meta, after_parts = _template_state(template)
            assert after_meta == before_meta
            for (storage, geometry, data), (new_storage, new_geometry, new_data) in zip(
                before_parts, after_parts
            ):
                assert storage._cdata == new_storage._cdata and geometry == new_geometry
                # Physical bytes also retain NaN payloads and signed zero.
                assert torch.equal(
                    data.contiguous().view(torch.uint8),
                    new_data.contiguous().view(torch.uint8),
                )
        except Exception as error:
            raise AssertionError("Candidate changed its read-only template") from error


def _sparse_coo(
    shape, sparse_dim, nnz, dtype, value_range, *, coalesced=True, payload=None
):
    """Sparse COO tensor holding ``nnz`` entries over ``shape``.

    Coordinates come from a seeded CPU permutation of the sparse coordinate
    space, so the tensor is well formed without calling ``coalesce()`` -- that
    kernel has no fp8 CUDA implementation and the values here may be fp8.
    """
    dense_shape = tuple(shape[sparse_dim:])
    if nnz == 0:
        indices = torch.empty(sparse_dim, 0, dtype=torch.long, device=flag_gems.device)
        payload = _to_target(
            torch.empty((0,) + dense_shape, dtype=dtype, device="cpu"), dtype
        )
    else:
        generator = torch.Generator("cpu").manual_seed(0)
        unique = nnz if coalesced else nnz - 1
        flat = torch.sort(
            torch.randperm(math.prod(shape[:sparse_dim]), generator=generator)[:unique]
        ).values
        if not coalesced:
            # Repeating the first coordinate keeps the tensor non-coalesced.
            flat = torch.cat([flat[:1], flat])
        indices = torch.stack(torch.unravel_index(flat, shape[:sparse_dim]), dim=0).to(
            flag_gems.device
        )
        payload = (
            _make_input(dtype, (nnz,) + dense_shape, value_range)
            if payload is None
            else _to_target(payload, dtype, (nnz,) + dense_shape)
        )
    result = torch.sparse_coo_tensor(
        indices, payload, shape, device=flag_gems.device, is_coalesced=coalesced
    )
    return _assert_input(result, shape, sparse_dim, nnz, dtype)


def _make_template(kind, shape, sparse_dim, dtype):
    """Template tensor; only its shape and sparse/dense split are read."""
    if kind == "dense":
        return _to_target(torch.zeros(shape, dtype=dtype, device="cpu"), dtype, shape)
    return _sparse_coo(shape, sparse_dim, 0, dtype, _DEFAULT_RANGE)


def _assert_resized(res, ref, *, original_values=None):
    _assert_devices(res, ref)
    assert res.layout == ref.layout
    assert res.shape == ref.shape
    assert res.dtype == ref.dtype
    assert res.sparse_dim() == ref.sparse_dim()
    assert res.dense_dim() == ref.dense_dim()
    assert res.is_coalesced() == ref.is_coalesced()
    assert res._indices().shape == ref._indices().shape
    assert res._values().shape == ref._values().shape
    _assert_values(res._indices(), ref._indices())
    if original_values is None:
        _assert_values(res._values(), ref._values())
    else:
        # Growing a dense extent reallocates the values block; the reference's
        # own new region is unspecified, so only the retained prefix is defined.
        n = original_values.numel()
        _assert_values(res._values(), ref._values(), n)
        _assert_values(_cpu_values(res._values(), n), original_values)


# Every row above is a cheap small-tensor branch (same size, sparse/dense
# growth, duplicate coordinates, empty, rank 0, multi-sparse and multi-dense
# splits), so the quick level keeps all of them; only the larger spec shapes and
# the positive special values stay default-only.
_RESIZE_CASES = tu.selected_cases(_ALL_CASES, quick=_ALL_CASES)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize("case", _RESIZE_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_resize_as_sparse_(case, dtype):
    (
        self_shape,
        self_sparse_dim,
        nnz,
        coalesced,
        kind,
        template_shape,
        template_sparse_dim,
        retains_storage,
        grows_dense,
    ) = case
    inp = _sparse_coo(
        self_shape, self_sparse_dim, nnz, dtype, _DEFAULT_RANGE, coalesced=coalesced
    )
    ref_inp = _reference(inp)
    template = _make_template(kind, template_shape, template_sparse_dim, dtype)
    ref_template = _reference(template)
    old_indices_storage = inp._indices().untyped_storage()
    old_indices_id = old_indices_storage._cdata
    old_indices_ptr = inp._indices().data_ptr()
    old_indices_extent = old_indices_storage.nbytes()
    old_values_storage = inp._values().untyped_storage()
    old_values_id = old_values_storage._cdata
    old_values_ptr = inp._values().data_ptr()
    old_values_extent = old_values_storage.nbytes()
    original_values = _cpu_values(inp._values()).flatten() if grows_dense else None

    _require_native_resize(inp, template)
    ref_out = torch.ops.aten.resize_as_sparse_(ref_inp, ref_template)
    res_out = _candidate_resize(inp, template)

    # In-place: the operator returns self and rewrites self's own metadata.
    assert res_out is inp
    assert inp.shape == ref_inp.shape
    if retains_storage:
        # A resize that keeps both extents and the split only rewrites metadata,
        # so the allocator keeps the original index and value blocks.
        assert res_out._indices().untyped_storage()._cdata == old_indices_id
        assert res_out._indices().data_ptr() == old_indices_ptr
        assert res_out._indices().untyped_storage().nbytes() == old_indices_extent
        assert res_out._values().untyped_storage()._cdata == old_values_id
        assert res_out._values().data_ptr() == old_values_ptr
        assert res_out._values().untyped_storage().nbytes() == old_values_extent
    _assert_resized(res_out, ref_out, original_values=original_values)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize("case", _RESIZE_CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_resize_as_sparse_value_ranges(case, dtype, value_range):
    (
        self_shape,
        self_sparse_dim,
        nnz,
        coalesced,
        kind,
        template_shape,
        template_sparse_dim,
        _retains_storage,
        grows_dense,
    ) = case
    inp = _sparse_coo(
        self_shape, self_sparse_dim, nnz, dtype, value_range, coalesced=coalesced
    )
    ref_inp = _reference(inp)
    template = _make_template(kind, template_shape, template_sparse_dim, dtype)
    ref_template = _reference(template)
    original_values = _cpu_values(inp._values()).flatten() if grows_dense else None

    _require_native_resize(inp, template)
    ref_out = torch.ops.aten.resize_as_sparse_(ref_inp, ref_template)
    res_out = _candidate_resize(inp, template)

    assert res_out is inp
    _assert_resized(res_out, ref_out, original_values=original_values)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_resize_as_sparse_shape_levels(shape, dtype, value_range):
    # A sparse COO tensor keeps sparse_dim + dense_dim == rank, so one sparse
    # dimension (none for a rank-0 input) keeps the dense tail; the template
    # grows only the outer sparse extent.
    sparse_dim = 1 if shape else 0
    nnz = min(2, shape[0]) if sparse_dim else 0
    template_shape = (shape[0] + 1,) + tuple(shape[1:]) if shape else ()
    inp = _sparse_coo(shape, sparse_dim, nnz, dtype, value_range)
    ref_inp = _reference(inp)
    template = _make_template("sparse", template_shape, sparse_dim, dtype)
    ref_template = _reference(template)

    _require_native_resize(inp, template)
    ref_out = torch.ops.aten.resize_as_sparse_(ref_inp, ref_template)
    res_out = _candidate_resize(inp, template)

    assert res_out is inp
    _assert_resized(res_out, ref_out)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[])
)
def test_resize_as_sparse_special_values(dtype, scenario):
    # nnz == 5 with no dense extent keeps the values block 1-D, so the shared
    # special-value payload feeds it directly.
    payload = _make_special_input(dtype, scenario)
    inp = _sparse_coo((4, 5), 2, 5, dtype, _DEFAULT_RANGE, payload=payload)
    ref_inp = _reference(inp)
    template = _make_template("sparse", (6, 5), 2, dtype)
    ref_template = _reference(template)

    _require_native_resize(inp, template)
    ref_out = torch.ops.aten.resize_as_sparse_(ref_inp, ref_template)
    res_out = _candidate_resize(inp, template)

    assert res_out is inp
    _assert_resized(res_out, ref_out)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize("case", _NEGATIVE_CASES)
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_resize_as_sparse_invalid_template(case, dtype):
    self_shape, self_sparse_dim, nnz, kind, template_shape, template_sparse_dim = case
    inp = _sparse_coo(self_shape, self_sparse_dim, nnz, dtype, _DEFAULT_RANGE)
    template = _make_template(kind, template_shape, template_sparse_dim, dtype)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _candidate_resize(inp, template)


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize("dtype", _NEGATIVE_DTYPES)
def test_resize_as_sparse_non_tensor_template(dtype):
    # The schema requires a Tensor template; a sequence must not be accepted.
    inp = _sparse_coo((4, 5), 2, 3, dtype, _DEFAULT_RANGE)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.resize_as_sparse_(inp, [6, 5])


@pytest.mark.resize_as_sparse_
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.complex64,
            torch.complex128,
        ],
        quick=[],
    ),
)
def test_resize_as_sparse_missing_derivative(dtype):
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64, device=flag_gems.device)
    values = _make_input(dtype, (2,), ["-1", "1"]).requires_grad_()
    inp = torch.sparse_coo_tensor(indices, values, (3, 4))
    template = _make_template("sparse", (5, 4), 2, dtype)
    res = _candidate_resize(inp, template)
    assert res is inp
    assert res.requires_grad
    grad = torch.sparse_coo_tensor(
        indices,
        _to_target(torch.ones(2, dtype=torch.float32, device="cpu").to(dtype), dtype),
        res.shape,
    )
    with pytest.raises(
        RuntimeError, match="derivative for aten::resize_as_sparse_ is not implemented"
    ):
        torch.autograd.grad(res, values, grad)
