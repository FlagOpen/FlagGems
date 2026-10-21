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


def _expected_device():
    device = torch.device(flag_gems.device)
    if device.index is None and device.type not in ("cpu", "meta"):
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return device


def _physical_cpu(inp):
    # Preserve independent physical storage, view geometry and lazy metadata.
    assert inp.layout == torch.strided
    raw = (
        torch.empty(0, dtype=torch.uint8, device=inp.device)
        .set_(inp.untyped_storage(), 0, (inp.untyped_storage().nbytes(),), (1,))
        .detach()
        .cpu()
        .clone()
    )
    result = torch.empty(0, dtype=inp.dtype, device="cpu").set_(
        raw.untyped_storage(), inp.storage_offset(), inp.size(), inp.stride()
    )
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    result.requires_grad_(inp.requires_grad)
    assert result.device == torch.device("cpu") and result.dtype == inp.dtype
    assert result.shape == inp.shape and result.stride() == inp.stride()
    assert result.storage_offset() == inp.storage_offset()
    assert (result.is_conj(), result.is_neg()) == (inp.is_conj(), inp.is_neg())
    if inp.untyped_storage().nbytes():
        assert result.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    return result


def _reference(inp):
    assert inp.device == _expected_device()
    return _physical_cpu(inp) if utils.TO_CPU else tu.to_reference(inp)


def _numeric_values(inp):
    result = _physical_cpu(inp.detach()).resolve_conj().resolve_neg()
    return result.float() if result.dtype in _FP8_DTYPES else result


def _assert_result_equal(result, reference):
    # Check the original ABI before observing numerical values on CPU.
    assert result.device == _expected_device()
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else result.device)
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape and result.layout == reference.layout
    if result.dtype in _FP8_DTYPES or (
        flag_gems.vendor_name in ("ascend", "kunlunxin")
        and (
            result.is_complex()
            or result.is_conj()
            or result.is_neg()
            or reference.is_conj()
            or reference.is_neg()
            or (
                flag_gems.vendor_name == "ascend"
                and result.dtype in (torch.int64, torch.float64)
            )
        )
    ):
        result, reference = _numeric_values(result), _numeric_values(reference)
    tu.assert_result_equal(result, reference)


def _check_requested_dtype(dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype in (
        torch.float64,
        torch.complex128,
    ):
        pytest.skip(
            "Kunlunxin tensor transfer narrows float64/complex128 to float32/complex64"
        )


def _cpu_encode(dtype):
    return flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES


def _make_input(dtype, shape, value_range):
    _check_requested_dtype(dtype)
    if _cpu_encode(dtype):
        low, high = (tu.resolve_bound(bound, dtype) for bound in value_range)
        if low == high:
            result = torch.full(shape, low, dtype=dtype, device="cpu")
        else:
            result = torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        result = result.to(flag_gems.device)
    else:
        result = tu.make_input(dtype, shape, value_range)
    assert result.dtype == dtype and result.shape == torch.Size(shape)
    assert result.device == _expected_device()
    return result


def _make_special_input(dtype, scenario):
    _check_requested_dtype(dtype)
    if _cpu_encode(dtype) or (
        flag_gems.vendor_name == "ascend" and dtype == torch.float64
    ):
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        result = (
            torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu")
            .to(dtype)
            .to(flag_gems.device)
        )
    else:
        result = tu.make_special_input(dtype, scenario)
    assert result.dtype == dtype and result.shape == (5,)
    assert result.device == _expected_device()
    return result


_DTYPE_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.complex128: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}

# aten::swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a) only permutes the
# size/stride metadata of an existing allocation, so every storable dtype (bool,
# integer, FP8, complex) is a valid operand and only the dims can be invalid.
_DTYPES = list(tu.REQUIRED_DTYPES) + [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _DTYPES += [torch.float64, torch.complex128]
_DTYPES = [dtype for dtype in _DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# Spec parameter-coverage dtypes, used for the dim-pair sweep.
_DIM_DTYPES = [
    torch.bfloat16,
    torch.float16,
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.int32,
    torch.int64,
]
_DIM_DTYPES = [dtype for dtype in _DIM_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# Value grid: one valid dim pair per required shape; rank 0/1 accept only identity.
_GRID_ROWS = [
    ((), 0, 0),
    ((1,), 0, 0),
    ((256,), 0, 0),
    ((1024, 1024), 0, 1),
    ((20, 320, 15), 0, 2),
    ((16, 128, 64, 60), 1, 3),
    ((16, 7, 57, 32, 29), 0, 4),
]
_GRID_CASES = tu.selected_cases(_GRID_ROWS, quick=[((2, 19, 7), 0, 2)])

# Dim sweep shapes: the quick shape plus 0-d, size-1, empty and large boundaries.
_DIM_SHAPES = [
    (2, 19, 7),
    (),
    (1,),
    (1, 1),
    (0, 3),
    (3, 0),
    (5, 0, 7),
    (1024, 1024),
    (20, 320, 15),
]


def _dim_pairs(shape):
    """One representative call per unordered dim pair, plus a negative pair."""
    rank = len(shape)
    if rank == 0:
        return [(0, 0)]
    pairs = []
    seen = set()
    for dim0 in range(rank):
        for dim1 in range(rank):
            key = tuple(sorted((dim0, dim1)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((dim0, dim1))
    if rank > 1:
        pairs.append((-1, -2))
    return pairs


def _dim_rows(shapes):
    return [(shape, dim0, dim1) for shape in shapes for dim0, dim1 in _dim_pairs(shape)]


_DIM_ROWS = tu.selected_cases(_dim_rows(_DIM_SHAPES), quick=_dim_rows(_DIM_SHAPES[:-2]))

# Views over stepped strides, storage offsets, 0-stride expansion and the
# small/empty boundaries; the swap must preserve all of them.
_LAYOUT_ROWS = [
    ("asis", (4, 12), 0, 1),
    ("column_step", (4, 24), 0, 1),
    ("row_step", (24, 8), 0, 1),
    ("transposed", (8, 16, 32), 0, 2),
    ("offset_window", (12, 40), 1, 0),
    ("expanded", (4, 3, 6), 0, 1),
    ("size1_axis", (1, 12), 0, 1),
    ("scalar", (), 0, 0),
    ("empty", (0, 3), 0, 1),
]
_LAYOUT_CASES = _LAYOUT_ROWS
_LAYOUT_DTYPES = [torch.float32, torch.int32, torch.bool]

# The lazy conjugate bit is metadata too, so it must survive the swap.
_CONJ_ROWS = [
    ((4, 12), 0, 1),
    ((12, 4), 1, 0),
]
_CONJ_DTYPES = [torch.complex64]
if utils.fp64_is_supported:
    _CONJ_DTYPES.append(torch.complex128)

_BACKWARD_ROWS = tu.selected_cases(
    [
        ((4, 12), 0, 1),
        ((2, 3, 5), 0, 2),
        ((2, 3, 5), 1, 2),
        ((2, 3, 5), -1, -2),
    ],
    quick=[],
)
_BACKWARD_DTYPES = [
    dtype for dtype in _DTYPES if dtype.is_floating_point or dtype.is_complex
]

# Out-of-range indices are the only invalid value a dim argument can take.
_OUT_OF_RANGE_ROWS = [
    ((4, 12), 0, 2),
    ((4, 12), 2, 0),
    ((4, 12), -3, 0),
    ((4, 12), 0, -3),
    ((2, 3, 4), 0, 3),
    ((0, 3), 0, 2),
    ((1, 1), 1, 2),
    ((256,), 0, 1),
    ((), 0, 1),
]
_NON_INTEGER_DIMS = [0.0, "0", [0]]


def _assert_view_facts(res, inp, ref, ref_inp):
    """The result is a fresh zero-copy alias of ``inp`` mirroring ``ref``."""
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    assert res.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res is not inp
    assert res._is_view()
    assert res.shape == ref.shape
    assert res.stride() == ref.stride()
    assert res.storage_offset() == inp.storage_offset()
    assert res.data_ptr() == inp.data_ptr()
    assert res.is_conj() == ref.is_conj()


def _layout_input(layout, shape, dtype):
    base = _make_input(dtype, shape, ["-1", "1"])
    if layout in ("asis", "size1_axis", "scalar", "empty"):
        return base
    if layout == "column_step":
        return base[:, ::2]
    if layout == "row_step":
        return base[::3]
    if layout == "transposed":
        return base.permute(2, 0, 1)
    if layout == "offset_window":
        return base[2:6, 3:35]
    if layout == "expanded":
        return base[0:1].expand(4, 3, 6)
    raise AssertionError(f"unknown layout {layout}")


def _swapped_shape(shape, dim0, dim1):
    rank = len(shape)
    perm = list(range(rank))
    perm[dim0 % rank], perm[dim1 % rank] = perm[dim1 % rank], perm[dim0 % rank]
    return tuple(shape[axis] for axis in perm)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.swapdims
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
@pytest.mark.parametrize("shape,dim0,dim1", _GRID_CASES)
def test_swapdims(shape, dim0, dim1, dtype, value_range):
    if flag_gems.vendor_name == "kunlunxin" and shape == ():
        pytest.skip("Kunlunxin scalar swapdims does not alias the input storage")
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.swapdims(ref_inp, dim0, dim1)
    res_out = flag_gems.swapdims(inp, dim0, dim1)

    _assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, inp, ref_out, ref_inp)


@pytest.mark.swapdims
@pytest.mark.parametrize("dtype", _DIM_DTYPES)
@pytest.mark.parametrize("shape,dim0,dim1", _DIM_ROWS)
def test_swapdims_dim_variants(shape, dim0, dim1, dtype):
    if flag_gems.vendor_name == "kunlunxin" and shape == ():
        pytest.skip("Kunlunxin scalar swapdims does not alias the input storage")
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.swapdims(ref_inp, dim0, dim1)
    res_out = flag_gems.swapdims(inp, dim0, dim1)

    _assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, inp, ref_out, ref_inp)


@pytest.mark.swapdims
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
@pytest.mark.parametrize("layout,shape,dim0,dim1", _LAYOUT_CASES)
def test_swapdims_layouts(layout, shape, dim0, dim1, dtype):
    if flag_gems.vendor_name == "kunlunxin" and shape == ():
        pytest.skip("Kunlunxin scalar swapdims does not alias the input storage")
    inp = _layout_input(layout, shape, dtype)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.swapdims(ref_inp, dim0, dim1)
    res_out = flag_gems.swapdims(inp, dim0, dim1)

    _assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, inp, ref_out, ref_inp)

    # Writing through the view must still reach the original storage.
    res_out.fill_(1)
    ref_out.fill_(1)
    _assert_result_equal(res_out, ref_out)
    _assert_result_equal(inp, ref_inp)


@pytest.mark.swapdims
@pytest.mark.parametrize("dtype", _CONJ_DTYPES)
@pytest.mark.parametrize("shape,dim0,dim1", _CONJ_ROWS)
def test_swapdims_conjugate_view(shape, dim0, dim1, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"]).conj()
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.swapdims(ref_inp, dim0, dim1)
    res_out = flag_gems.swapdims(inp, dim0, dim1)

    _assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, inp, ref_out, ref_inp)
    assert res_out.is_conj()


# `tu.special_value_cases` already drops inf-bearing scenarios for float8_e4m3fn,
# which cannot represent infinity.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[])


@pytest.mark.swapdims
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_swapdims_special_values(dtype, scenario):
    inp = _make_special_input(dtype, scenario).reshape(1, 5)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.swapdims(ref_inp, 0, 1)
    res_out = flag_gems.swapdims(inp, 0, 1)

    _assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, inp, ref_out, ref_inp)


@pytest.mark.swapdims
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
@pytest.mark.parametrize("shape,dim0,dim1", _BACKWARD_ROWS)
def test_swapdims_backward(shape, dim0, dim1, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = _reference(inp).requires_grad_(True)
    upstream = _make_input(dtype, _swapped_shape(shape, dim0, dim1), ["-1", "1"])
    ref_upstream = _reference(upstream)

    ref_out = torch.ops.aten.swapdims(ref_inp, dim0, dim1)
    res_out = flag_gems.swapdims(inp, dim0, dim1)

    _assert_result_equal(res_out, ref_out)
    _assert_view_facts(res_out, inp, ref_out, ref_inp)

    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=upstream)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)

    _assert_result_equal(res_grad, ref_grad)


@pytest.mark.swapdims
@pytest.mark.parametrize("shape,dim0,dim1", _OUT_OF_RANGE_ROWS)
def test_swapdims_invalid_dims(shape, dim0, dim1):
    if flag_gems.vendor_name == "kunlunxin" and (shape, dim0, dim1) in (
        ((4, 12), -3, 0),
        ((4, 12), 0, -3),
        ((2, 3, 4), 0, 3),
        ((256,), 0, 1),
        ((), 0, 1),
    ):
        pytest.skip("Kunlunxin swapdims cannot safely reject these out-of-range dims")
    inp = _make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems.swapdims(inp, dim0, dim1)


@pytest.mark.swapdims
@pytest.mark.parametrize("bad_dim", _NON_INTEGER_DIMS)
def test_swapdims_non_integer_dims(bad_dim):
    inp = _make_input(torch.float32, (4, 12), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.swapdims(inp, bad_dim, 0)
