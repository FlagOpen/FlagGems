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


def _make_input(dtype, shape, value_range):
    if dtype not in _FP8_DTYPES or flag_gems.vendor_name not in ("ascend", "kunlunxin"):
        return tu.make_input(dtype, shape, value_range)
    low, high = (tu.resolve_bound(v, dtype) for v in value_range)
    if low == high:
        value = torch.full(shape, low, dtype=torch.float32, device="cpu")
    else:
        value = torch.testing.make_tensor(
            shape, dtype=torch.float32, device="cpu", low=low, high=high
        )
    return value.to(dtype).to(flag_gems.device)


def _reference(inp, *, cpu=None):
    # Copy storage as bytes before rebuilding the view. A Tensor.cpu() copy
    # compacts nonzero offsets; typed FP8 strided copies are also unavailable
    # on some backends. The native metadata query still runs on an independent
    # tensor with the same shape, strides, offset and lazy flags.
    if inp.device.type == "meta":
        return inp.detach().clone()
    if inp.layout != torch.strided:
        return tu.to_reference(inp)
    if cpu is None:
        cpu = utils.TO_CPU
    nbytes = inp.untyped_storage().nbytes()
    raw = torch.empty(0, dtype=torch.uint8, device=inp.device).set_(
        inp.untyped_storage(), 0, (nbytes,), (1,)
    )
    raw = raw.cpu().clone() if cpu else raw.clone()
    ref = torch.empty(0, dtype=inp.dtype, device=raw.device).set_(
        raw.untyped_storage(), inp.storage_offset(), inp.size(), inp.stride()
    )
    if inp.is_conj():
        ref = ref.conj()
    if inp.is_neg():
        ref = ref._neg_view()
    return ref.requires_grad_(inp.requires_grad)


def _assert_unchanged(actual, snapshot):
    assert actual.device == snapshot.device
    assert actual.dtype == snapshot.dtype and actual.shape == snapshot.shape
    observed, expected = _reference(actual, cpu=True), _reference(snapshot, cpu=True)
    if actual.dtype in _FP8_DTYPES:
        observed, expected = observed.float(), expected.float()
    torch.testing.assert_close(observed, expected, rtol=0, atol=0, equal_nan=True)


# storage_offset reads view metadata and returns a Python int. The shared value
# ranges verify that payloads do not affect the answer; layout/view chains vary
# the actual offset. There is no binary broadcast or differentiable result.
_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.bool, torch.complex64]
)

# (layout, minimum rank). Each entry is a real view chain whose offset follows
# from stride arithmetic.
_CORE_LAYOUTS = (
    ("contiguous", 0),
    ("as_strided", 0),  # explicit storage_offset on a flat view
    ("slice_rows", 1),  # base[2:]
    ("chained_rows", 1),  # base[3:][1:]
    ("narrow_rows", 1),  # base.narrow(0, start, rest)
    ("expanded", 1),  # stride-0 view of base[1:2]
)
_STRIDED_LAYOUTS = (
    ("slice_cols", 2),  # base[:, 1:]
    ("select_row", 2),  # base[1]
    ("transpose", 2),  # offset unchanged while the strides swap
    ("transpose_rows", 2),  # base.transpose(-1, -2)[1:]
    ("diagonal", 2),  # diagonal(offset=1) over the first two dims
    ("unfold", 2),  # base[1:].unfold(0, step, 1)
)


def _offset_view(base, layout):
    """Return a view of ``base`` whose storage offset comes from ``layout``."""
    shape = base.shape
    if layout == "contiguous":
        return base
    if layout == "as_strided":
        flat = base.reshape(-1)
        start = min(3, flat.numel())
        return torch.as_strided(flat, (flat.numel() - start,), (1,), start)
    if layout == "slice_rows":
        return base[2:]
    if layout == "chained_rows":
        return base[3:][1:]
    if layout == "narrow_rows":
        start = min(1, shape[0])
        return base.narrow(0, start, shape[0] - start)
    if layout == "expanded":
        rows = base[1:2] if shape[0] >= 2 else base[:1]
        return rows.expand(3, *shape[1:])
    if layout == "slice_cols":
        return base[:, 1:]
    if layout == "select_row":
        return base[1]
    if layout == "transpose":
        return base.transpose(-1, -2)
    if layout == "transpose_rows":
        return base.transpose(-1, -2)[1:]
    if layout == "diagonal":
        return base.diagonal(offset=1)
    if layout == "unfold":
        tail = base[1:]
        return tail.unfold(0, min(2, tail.shape[0]), 1)
    raise ValueError(f"unknown layout: {layout!r}")


def _layout_rows(shapes, layouts):
    return [
        (shape, layout)
        for shape in shapes
        for layout, min_rank in layouts
        if len(shape) >= min_rank
    ]


# Each layout is constructed from a full shared shape before querying its offset.
_VIEW_ROWS = _layout_rows(tu.selected_shapes(), _CORE_LAYOUTS) + _layout_rows(
    [shape for shape in tu.selected_shapes() if len(shape) >= 2], _STRIDED_LAYOUTS
)


# Explicit default/quick pairs; dtype lists retain device capability filtering.
_VIEW_DTYPE_CASES = [
    (shape, layout, torch.float32) for shape, layout in _VIEW_ROWS
] + tu.selected_cases(
    [
        ((20, 320, 15), "slice_cols", dtype)
        for dtype in _DTYPES
        if dtype != torch.float32
    ],
    quick=[
        ((2, 19, 7), "slice_cols", dtype) for dtype in _DTYPES if dtype != torch.float32
    ],
)


@pytest.mark.storage_offset
@pytest.mark.parametrize("shape, layout, dtype", _VIEW_DTYPE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges()[:1])
def test_storage_offset_views(shape, layout, value_range, dtype):
    base = _make_input(dtype, shape, value_range)
    inp = _offset_view(base, layout)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.storage_offset(ref_inp)
    res_out = flag_gems.storage_offset(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out


# Scalars, empty tensors and views whose offset falls past the last element.
_SCALAR_EMPTY_SHAPES = [(), (0,), (1,), (0, 3), (3, 0), (5, 0, 7), (0, 0, 0)]
_SCALAR_EMPTY_DTYPES = [torch.float32, torch.int64, torch.float8_e4m3fn, torch.bool]
_SCALAR_EMPTY_ROWS = _layout_rows(
    _SCALAR_EMPTY_SHAPES,
    (("contiguous", 0), ("as_strided", 0), ("slice_rows", 1), ("chained_rows", 1)),
)


@pytest.mark.storage_offset
@pytest.mark.parametrize("shape,layout", _SCALAR_EMPTY_ROWS)
@pytest.mark.parametrize("dtype", _SCALAR_EMPTY_DTYPES)
def test_storage_offset_scalar_and_empty(shape, layout, dtype):
    base = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    inp = _offset_view(base, layout)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.storage_offset(ref_inp)
    res_out = flag_gems.storage_offset(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out
    if shape == () and layout == "contiguous":
        # A scalar tensor still owns one element of storage at offset zero.
        assert res_out == 0


_CHAIN_STEPS = ("base", "slice_rows", "slice_twice", "narrow", "transposed")
_CHAIN_DTYPES = [
    torch.float32,
    torch.float8_e5m2,
    torch.int8,
    torch.float64,
    torch.bool,
]


def _chain_view(base, step):
    """Views of one storage with increasing offsets, plus one transposed view."""
    if step == "base":
        return base
    if step == "slice_rows":
        return base[3:]
    if step == "slice_twice":
        return base[3:][2:]
    if step == "narrow":
        return base[3:].narrow(0, 2, 6)
    if step == "transposed":
        return base[5].transpose(0, 1)[2:]
    raise ValueError(f"unknown chain step: {step!r}")


@pytest.mark.storage_offset
@pytest.mark.parametrize("step", _CHAIN_STEPS)
@pytest.mark.parametrize("dtype", _CHAIN_DTYPES)
def test_storage_offset_view_chain(step, dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip(
            "Kunlunxin cannot create float64 storage without narrowing to float32"
        )
    base = torch.empty((16, 128, 64), dtype=dtype, device=flag_gems.device)
    assert base.dtype == dtype
    inp = _chain_view(base, step)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.storage_offset(ref_inp)
    res_out = flag_gems.storage_offset(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out
    if step == "base":
        assert res_out == 0
    else:
        assert res_out > 0


_LAZY_VIEW_CASES = [("conj", torch.complex64), ("neg", torch.float32)]


@pytest.mark.storage_offset
@pytest.mark.parametrize("kind,dtype", _LAZY_VIEW_CASES)
def test_storage_offset_lazy_bit_views(kind, dtype):
    # A view that already carries a lazy conjugate/negative bit reports the same
    # offset as the plain slice it was taken from.
    base = _make_input(dtype, (16, 128, 64), ["-1", "1"])[5:]
    inp = base.conj() if kind == "conj" else base._neg_view()
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.storage_offset(ref_inp)
    res_out = flag_gems.storage_offset(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out


@pytest.mark.storage_offset
def test_storage_offset_aliasing_views_are_ordered():
    # Two views of one storage: their reported offsets must differ by exactly the
    # element distance between their starts, which a per-view comparison against
    # the reference cannot establish on its own.
    base = torch.empty((16, 128, 64), dtype=torch.float32, device=flag_gems.device)
    left = base[3:]
    right = base[5:]

    left_ref = torch.ops.aten.storage_offset(_reference(left))
    right_ref = torch.ops.aten.storage_offset(_reference(right))
    left_res = flag_gems.storage_offset(left)
    right_res = flag_gems.storage_offset(right)

    assert type(left_res) is int and type(right_res) is int
    assert left_res == left_ref
    assert right_res == right_ref
    assert right_res - left_res == 2 * base.stride(0)


@pytest.mark.storage_offset
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.complex64])
def test_storage_offset_leaves_input_state_unchanged(dtype):
    inp = _make_input(dtype, (20, 320, 15), ["-1", "1"])
    view = inp[4:9].transpose(1, 2)
    before = view.clone()
    stride_before = view.stride()
    ref_view = _reference(view)

    ref_out = torch.ops.aten.storage_offset(ref_view)
    res_out = flag_gems.storage_offset(view)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out
    # A metadata query must not copy the view's input: the aliased storage,
    # the strides and the values all survive the call.
    assert view.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert view.stride() == stride_before
    _assert_unchanged(view, before)


@pytest.mark.storage_offset
@pytest.mark.parametrize("dtype", _DTYPES)
def test_storage_offset_result_type_and_stability(dtype):
    inp = _make_input(dtype, (16, 128, 64), ["-1", "1"])[7:]
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.storage_offset(ref_inp)
    res_out = flag_gems.storage_offset(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out
    assert res_out > 0
    # Re-reading the same view must not move.
    assert flag_gems.storage_offset(inp) == res_out


# Sparse COO/CSR and meta tensors dispatch to the same metadata query; the meta
# tensor is created device-less on purpose because that is the layout under
# test, not the compute device.
@pytest.mark.storage_offset
@pytest.mark.parametrize("layout", ["coo", "csr", "meta"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_storage_offset_dispatch_layouts(layout, dtype):
    if layout == "coo":
        inp = _make_input(dtype, (8, 8), ["-1", "1"]).to_sparse()
    elif layout == "csr":
        inp = _make_input(dtype, (8, 8), ["-1", "1"]).to_sparse_csr()
    else:
        inp = torch.empty((8, 8), dtype=dtype, device="meta")
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.storage_offset(ref_inp)
    res_out = flag_gems.storage_offset(inp)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out


def _special_input(shape, dtype, scenario):
    """Reshape the shared special-value payload to ``shape`` unchanged."""
    numel = 1
    for dim in shape:
        numel *= dim
    if dtype in _FP8_DTYPES and flag_gems.vendor_name in ("ascend", "kunlunxin"):
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        values = torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu").to(
            dtype
        )
        repeats = (numel + values.numel() - 1) // values.numel()
        return values.repeat(repeats)[:numel].reshape(shape).to(flag_gems.device)
    values = tu.make_special_input(dtype, scenario)
    repeats = (numel + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:numel].reshape(shape)


# The query never reads elements, so NaN/Inf payloads only exercise the dtype x
# scenario matrix of the shared generator; this test stays default-only.
_SPECIAL_SHAPES = tu.selected_cases([(), (5,), (4, 3)], quick=[])
_SPECIAL_ROWS = [
    (shape, layout)
    for shape in _SPECIAL_SHAPES
    for layout in ("base", "slice_rows")
    if layout == "base" or len(shape) >= 1
]


@pytest.mark.storage_offset
@pytest.mark.parametrize("shape,layout", _SPECIAL_ROWS)
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[])
)
def test_storage_offset_special_values(shape, layout, dtype, scenario):
    inp = _special_input(shape, dtype, scenario)
    view = inp if layout == "base" else inp[1:]
    ref_view = _reference(view)

    ref_out = torch.ops.aten.storage_offset(ref_view)
    res_out = flag_gems.storage_offset(view)

    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out


# The schema takes exactly one Tensor. These are the argument forms the native
# dispatcher rejects; only the candidate's exception is asserted. AttributeError
# is deliberately not accepted, so a missing candidate cannot pass as an
# invalid input.
_INVALID_ARGS = [
    pytest.param((), id="no_args"),
    pytest.param((3.14,), id="float_self"),
    pytest.param((1,), id="int_self"),
    pytest.param(("x",), id="str_self"),
    pytest.param(([1.0, 2.0],), id="list_self"),
    pytest.param(((1, 2),), id="tuple_self"),
]


@pytest.mark.storage_offset
@pytest.mark.parametrize("args", _INVALID_ARGS)
def test_storage_offset_rejects_invalid_arguments(args):
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.storage_offset(*args)


@pytest.mark.storage_offset
def test_storage_offset_rejects_extra_argument():
    inp = torch.empty((4,), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.storage_offset(inp, inp)
