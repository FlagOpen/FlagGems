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


# aten::stride is a host-side metadata query: it has no broadcast form and no
# autograd formula, so those spec dimensions do not apply. Coverage uses both
# callable overloads, layouts with non-unit or zero strides and non-zero storage
# offsets, empty and 0-dim tensors, and invalid dimensions.
# aten::stride.Dimname needs a named tensor, and every torch.ops.aten call
# written with names raises "NYI: Named tensors are currently unsupported in
# TorchScript", so that overload has no native reference.
_STRIDE_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.int16, torch.bool]
)

_EXPANDED_SHAPE = (4, 8, 16)

# Each row is cheap and reports a distinct stride pattern, so all rows stay in
# the quick subset too.
_LAYOUT_ROWS = [
    ((4, 6), "contiguous"),
    ((4, 6), "transposed"),
    ((4, 6), "step"),
    ((4, 6), "narrow"),
    ((4, 6), "sliced"),
    ((4, 6), "permuted"),
    ((1, 8, 16), "expanded"),
    ((256,), "as_strided"),
    ((2, 3, 4, 5), "channels_last"),
]

# Layouts only vary how storage is addressed, so one float, one integer and one
# fp8 type suffice; the full dtype set runs in the value grid.
_LAYOUT_DTYPES = [torch.float32, torch.int64, torch.float8_e4m3fn]

_EMPTY_SHAPES = [(0,), (0, 3), (2, 0, 3), (3, 0, 0)]

_SPARSE_CASES = [((4, 4), 8), ((8, 8, 8), 64), ((3, 5, 7), 17)]

_SPECIAL_SHAPES = [(5,), (1, 5)]

# Out-of-range dimensions: one dim past either end of the valid range.
_OOB_SHAPES = [(4, 6), (2, 3, 4)]


def _apply_layout(base, layout):
    if layout == "contiguous":
        return base
    if layout == "transposed":
        return base.transpose(0, -1)
    if layout == "step":
        return base[:, ::3]
    if layout == "narrow":
        return base.narrow(0, 1, 2)
    if layout == "sliced":
        return base[1:, ::2]
    if layout == "permuted":
        return base.permute(1, 0)
    if layout == "expanded":
        return base.expand(_EXPANDED_SHAPE)
    if layout == "as_strided":
        return torch.as_strided(base, (3, 4), (7, 2), 5)
    if layout == "channels_last":
        if base.device.type != "cpu" and (
            flag_gems.vendor_name == "ascend"
            or (
                flag_gems.vendor_name == "kunlunxin"
                and base.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            )
        ):
            host = _copy_to(base, "cpu").to(memory_format=torch.channels_last)
            return _copy_to(host, base.device)
        return base.to(memory_format=torch.channels_last)
    raise AssertionError("unknown layout: " + str(layout))


def _dim_cases_for(shapes):
    cases = []
    for shape in shapes:
        rank = len(shape)
        if rank == 0:
            continue  # a 0-dim tensor has no dimension to query
        dims = {0, -1, rank - 1, -rank}
        if rank > 2:
            dims |= {1, -2}
        cases.extend((shape, dim) for dim in sorted(dims))
    return cases


# `dim` parameter coverage: zero, positive, negative and both dimension bounds.
_DIM_CASES = tu.selected_cases(
    _dim_cases_for(list(dict.fromkeys(tu.REQUIRED_SHAPES + tu.QUICK_SHAPES))),
    quick=_dim_cases_for(tu.QUICK_SHAPES),
)


def _assert_stride_list(res_out, ref_out):
    # schema: stride(Tensor self) -> int[] (a sequence of Python ints).
    assert isinstance(res_out, (list, tuple)), type(res_out)
    assert all(type(item) is int for item in res_out), res_out
    assert list(res_out) == list(ref_out)


def _assert_stride_int(res_out, ref_out):
    # schema: stride.int(Tensor self, int dim) -> int
    assert type(res_out) is int, type(res_out)
    assert res_out == ref_out


def _layout_facts(tensor):
    # The query must not disturb the operand it reads.
    return (
        tuple(tensor.size()),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.data_ptr(),
        tensor.untyped_storage()._cdata,
        tensor.dtype,
        tensor.device,
    )


def _make_coo(shape, nnz, dtype):
    # Seeded CPU indices; repeats keep the tensor uncoalesced.
    generator = torch.Generator("cpu").manual_seed(0)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=generator)
            for dim in shape
        ]
    )
    values = _make_input(dtype, (nnz,), ["-1", "1"])
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _STRIDE_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.stride
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test_stride_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    before = _layout_facts(inp)

    ref_out = torch.ops.aten.stride(ref_inp)
    res_out = flag_gems.stride(inp)

    _assert_stride_list(res_out, ref_out)
    assert _layout_facts(inp) == before


@pytest.mark.stride
@pytest.mark.parametrize("shape,dim", _DIM_CASES)
@pytest.mark.parametrize("dtype", _STRIDE_DTYPES)
def test_stride_dim_over_shapes(shape, dim, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    before = _layout_facts(inp)

    ref_out = torch.ops.aten.stride(ref_inp, dim)
    res_out = flag_gems.stride(inp, dim)

    _assert_stride_int(res_out, ref_out)
    assert _layout_facts(inp) == before


@pytest.mark.stride
@pytest.mark.parametrize("storage_shape,layout", _LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_stride_layouts(storage_shape, layout, dtype):
    base = _make_input(dtype, storage_shape, ["-1", "1"])
    inp = _apply_layout(base, layout)
    ref_inp = _apply_layout(tu.to_reference(base), layout)
    before = _layout_facts(inp)

    ref_out = torch.ops.aten.stride(ref_inp)
    res_out = flag_gems.stride(inp)

    _assert_stride_list(res_out, ref_out)
    assert _layout_facts(inp) == before


@pytest.mark.stride
@pytest.mark.parametrize("storage_shape,layout", _LAYOUT_ROWS)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_stride_layout_dim(storage_shape, layout, dim, dtype):
    base = _make_input(dtype, storage_shape, ["-1", "1"])
    inp = _apply_layout(base, layout)
    ref_inp = _apply_layout(tu.to_reference(base), layout)

    ref_out = torch.ops.aten.stride(ref_inp, dim)
    res_out = flag_gems.stride(inp, dim)

    _assert_stride_int(res_out, ref_out)


@pytest.mark.stride
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
@pytest.mark.parametrize("call_args", [(), (0,)])
def test_stride_empty_shapes(shape, dtype, call_args):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.stride(ref_inp, *call_args)
    res_out = flag_gems.stride(inp, *call_args)
    if call_args:
        _assert_stride_int(res_out, ref_out)
    else:
        _assert_stride_list(res_out, ref_out)


@pytest.mark.stride
@pytest.mark.parametrize("shape,nnz", _SPARSE_CASES)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
@pytest.mark.parametrize("call_args", [(), (0,)])
def test_stride_sparse_coo(shape, nnz, dtype, call_args):
    inp = _make_coo(shape, nnz, dtype)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.stride(ref_inp, *call_args)
    res_out = flag_gems.stride(inp, *call_args)
    if call_args:
        _assert_stride_int(res_out, ref_out)
    else:
        _assert_stride_list(res_out, ref_out)


@pytest.mark.stride
@pytest.mark.parametrize("shape", _SPECIAL_SHAPES)
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_STRIDE_DTYPES))
)
@pytest.mark.parametrize("call_args", [(), (0,)])
def test_stride_special_values(dtype, scenario, shape, call_args):
    inp = _make_special_input(dtype, scenario).reshape(shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.stride(ref_inp, *call_args)
    res_out = flag_gems.stride(inp, *call_args)
    if call_args:
        _assert_stride_int(res_out, ref_out)
    else:
        _assert_stride_list(res_out, ref_out)


@pytest.mark.stride
@pytest.mark.parametrize("shape", _OOB_SHAPES)
@pytest.mark.parametrize("side", ["before_first", "after_last"])
def test_stride_rejects_out_of_range_dim(shape, side):
    inp = _make_input(torch.float32, shape, ["-1", "1"])
    dim = -len(shape) - 1 if side == "before_first" else len(shape)
    with pytest.raises((IndexError, RuntimeError, ValueError, TypeError)):
        flag_gems.stride(inp, dim)


@pytest.mark.stride
def test_stride_rejects_dim_on_scalar_tensor():
    inp = _make_input(torch.float32, (), ["-1", "1"])

    with pytest.raises((IndexError, RuntimeError, ValueError, TypeError)):
        flag_gems.stride(inp, 0)


@pytest.mark.stride
def test_stride_rejects_non_tensor():
    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems.stride(3.14)


@pytest.mark.stride
def test_stride_rejects_non_integer_dim():
    inp = _make_input(torch.float32, (4, 6), ["-1", "1"])

    with pytest.raises((TypeError, ValueError, RuntimeError, NotImplementedError)):
        flag_gems.stride(inp, 1.0)


@pytest.mark.stride
def test_stride_rejects_sparse_csr():
    inp = torch.sparse_csr_tensor(
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        torch.randn(2, device=flag_gems.device),
        (2, 2),
        device=flag_gems.device,
    )

    # Native raises RuntimeError("Sparse CSR tensors do not have strides").
    with pytest.raises((RuntimeError, NotImplementedError)):
        flag_gems.stride(inp)
