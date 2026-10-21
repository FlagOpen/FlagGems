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


def _cpu_storage_view(inp):
    # Copy physical bytes, retaining the original geometry before decoding.
    assert inp.layout == torch.strided
    raw = (
        torch.empty(0, dtype=torch.uint8, device=inp.device)
        .set_(inp.untyped_storage(), 0, (inp.untyped_storage().nbytes(),), (1,))
        .detach()
        .cpu()
        .clone()
    )
    return torch.empty(0, dtype=inp.dtype, device="cpu").set_(
        raw.untyped_storage(), inp.storage_offset(), inp.size(), inp.stride()
    )


def _numeric_values(inp):
    result = _cpu_storage_view(inp)
    # Decode before resolving lazy bits: CPU FP8/C32 arithmetic is limited.
    if result.dtype in _FP8_DTYPES:
        result = result.float()
    if result.dtype == torch.complex32:
        result = result.to(torch.complex64)
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    return result.resolve_conj().resolve_neg()


def _assert_result_equal(result, reference, *, required_device):
    # Validate the original ABI before observing numerical values on CPU.
    assert result.device == required_device
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else result.device)
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape and result.layout == reference.layout
    if (
        result.dtype in _FP8_DTYPES
        or result.dtype == torch.complex32
        or (flag_gems.vendor_name in ("ascend", "kunlunxin") and result.is_complex())
        or (flag_gems.vendor_name == "ascend" and result.dtype == torch.int64)
        or (flag_gems.vendor_name == "iluvatar" and result.dtype == torch.float64)
        or (
            flag_gems.vendor_name == "kunlunxin"
            and (result.is_neg() or reference.is_neg())
        )
    ):
        result, reference = _numeric_values(result), _numeric_values(reference)
    tu.assert_result_equal(result, reference)


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES:
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
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES:
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


def _reference(inp):
    assert inp.device == _expected_device()
    if not utils.TO_CPU:
        return tu.to_reference(inp)
    result = _cpu_storage_view(inp)
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    result.requires_grad_(inp.requires_grad)
    assert result.dtype == inp.dtype and result.shape == inp.shape
    assert (
        result.stride() == inp.stride()
        and result.storage_offset() == inp.storage_offset()
    )
    assert (result.is_conj(), result.is_neg()) == (inp.is_conj(), inp.is_neg())
    if inp.untyped_storage().nbytes():
        assert result.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    return result


def _fill_value(result, value):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and result.device.type != "cpu"
        and result.is_complex()
    ):
        result.copy_(
            torch.full(tuple(result.shape), value, dtype=result.dtype, device="cpu")
        )
    else:
        result.fill_(value)


# aten::matrix_H is the (conjugate) transpose view (Tensor(a) -> Tensor(a)): it
# swaps the last two dimensions and, for complex input, toggles the lazy
# conjugate bit instead of materialising a conjugate copy, so the tested
# contract is layout plus aliasing. Rank 2 is the supported class; rank 0 is a
# deprecated scalar path and rank 1/3+ raise, which drives the negative cases.
# No element is read, so every dtype the operator accepts is covered here.


def _supported_dtypes():
    dtypes = [
        dtype
        for dtype in tu.REQUIRED_DTYPES
        if utils.fp8_is_supported
        or dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
    ]
    if utils.fp64_is_supported:
        dtypes += [torch.float64, torch.complex128]
    return dtypes + [torch.bool, torch.complex64]


_MATRIX_H_DTYPES = _supported_dtypes()
_COMPLEX_DTYPES = [torch.complex64] + (
    [torch.complex128] if utils.fp64_is_supported else []
)

# Only matrices and the deprecated scalar form are valid positive inputs.
_MATRIX_H_SHAPES = tu.selected_cases(
    [(), (1, 1), (2, 19), (16, 16), (1024, 1024), (20, 320), (256, 8192), (8192, 32)],
    quick=[(), (1, 1), (2, 19)],
)

_LAYOUT_CASES = [
    # kind, shape, dtype: strides, offset and expansion are layout, not values.
    ("transposed", (4, 6), torch.float32),
    ("transposed", (7, 3), torch.bfloat16),
    ("column_slice", (4, 6), torch.float32),
    ("column_slice", (8, 5), torch.float16),
    ("offset_window", (6, 8), torch.float32),
    ("offset_window", (5, 7), torch.complex64),
    ("expanded", (4, 6), torch.float32),
    ("expanded", (3, 5), torch.int64),
    ("empty", (0, 0), torch.float32),
    ("empty", (3, 0), torch.int32),
    ("empty", (0, 1), torch.complex64),
]

_CONJUGATE_CASES = [
    ("plain", (4, 6)),
    ("column_slice", (4, 8)),
]

_WRITE_CASES = [
    ((4, 6), torch.float32),
    ((8, 3), torch.float16),
    ((5, 7), torch.complex64),
]

_BACKWARD_CASES = [
    ((4, 6), torch.float32),
    ((7, 13), torch.float16),
    ((16, 32), torch.bfloat16),
]
if utils.fp64_is_supported:
    _BACKWARD_CASES.append(((8, 8), torch.float64))
_BACKWARD_CASES += [
    ((4, 6), dtype)
    for dtype in _MATRIX_H_DTYPES
    if (dtype.is_floating_point or dtype.is_complex)
    and dtype not in {row[1] for row in _BACKWARD_CASES}
]

_SPECIAL_VALUE_CASES = tu.special_value_cases(_MATRIX_H_DTYPES)

# tu.special_value_cases covers floating dtypes only (is_floating_point is
# False for complex), so the complex payloads are spelled out here.
_COMPLEX_SPECIAL_PAYLOADS = {
    "nan": [complex(float("nan"), 0.0), complex(0.0, float("nan")), 1.0 - 1.0j],
    "inf": [complex(float("inf"), float("-inf")), 1.0j, -1.0 + 0.0j],
    "mixed": [complex(float("nan"), float("inf")), complex(float("-inf"), 0.0), 0j],
}

_INVALID_RANK_SHAPES = [(1,), (2, 3, 5), (2, 3, 4, 6), (16, 7, 57, 32, 29)]

_OVERLOAD_SHAPES = tu.selected_cases(
    [(4, 6), (64, 64), (1, 4096), (256, 8192)],
    quick=[(4, 6)],
)


def _assert_view_semantics(res_out, ref_out, inp, ref_inp):
    assert (res_out is inp) == (ref_out is ref_inp)
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out._is_view() == ref_out._is_view()
    # A view keeps addressing the very storage of its input.
    assert res_out.data_ptr() == inp.data_ptr()
    assert res_out.untyped_storage()._cdata == inp.untyped_storage()._cdata


def _layout_input(kind, shape, dtype):
    """Build the original layout; _reference preserves it in both ref modes."""
    inp = _make_input(dtype, shape, ["-1", "1"])
    if kind == "transposed":
        inp = inp.t()
    elif kind == "column_slice":
        inp = inp[:, ::2]
    elif kind == "offset_window":
        base = _make_input(dtype, (2 * shape[0], 2 * shape[1]), ["-1", "1"])
        inp = base[2 : 2 + shape[0], 1 : 1 + shape[1]]
    elif kind == "expanded":
        inp = _make_input(dtype, (1, shape[1]), ["-1", "1"]).expand(shape)
    elif kind != "empty":
        raise AssertionError(f"unknown layout {kind!r}")
    return inp


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _MATRIX_H_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.matrix_H
@pytest.mark.parametrize("shape", _MATRIX_H_SHAPES)
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test_matrix_H(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    required_device = inp.device
    res_out = flag_gems.matrix_H(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize(
    "kind,shape,dtype",
    _LAYOUT_CASES,
)
def test_matrix_H_layout(kind, shape, dtype):
    inp = _layout_input(kind, shape, dtype)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    required_device = inp.device
    res_out = flag_gems.matrix_H(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize("form,shape", _CONJUGATE_CASES)
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_matrix_H_conjugate_state(form, shape, dtype):
    # The lazy conjugate bit is this operator's primary semantic, so both the
    # complex dtypes and the conjugate rows are collected in every mode.
    # An already-set lazy conjugate bit must be toggled, not materialised.
    inp = _make_input(dtype, shape, ["-1", "1"])
    if form == "column_slice":
        inp = inp[:, ::2]
    inp = inp.conj()
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    required_device = inp.device
    res_out = flag_gems.matrix_H(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize(
    "shape,dtype",
    _WRITE_CASES,
)
def test_matrix_H_writes_through(shape, dtype):
    # The result addresses the input storage, so a write through it has to be
    # visible in the input (a materialised copy would silently drop the write).
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    required_device = inp.device
    res_out = flag_gems.matrix_H(inp)
    _fill_value(res_out, 2.5)
    _fill_value(ref_out, 2.5)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_result_equal(inp, ref_inp, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize("shape,dtype", tu.selected_cases(_BACKWARD_CASES, quick=[]))
def test_matrix_H_backward(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = _make_input(dtype, (shape[1], shape[0]), ["-1", "1"])
    ref_inp = _reference(inp)
    ref_grad = _reference(grad)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    required_device = inp.device
    res_out = flag_gems.matrix_H(inp)
    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]

    _assert_result_equal(res_in_grad, ref_in_grad, required_device=required_device)


@pytest.mark.matrix_H
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(_SPECIAL_VALUE_CASES, quick=[])
)
def test_matrix_H_special_values(dtype, scenario):
    # make_special_input emits a 1-D payload; matrix_H requires rank 2, so the
    # payload is reshaped without touching its values.
    inp = _make_special_input(dtype, scenario).reshape(5, 1)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    required_device = inp.device
    res_out = flag_gems.matrix_H(inp)

    # A view passes NaN/Inf through unchanged, so an exact comparison with
    # equal_nan matching is the right oracle.
    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize(
    "scenario", tu.selected_cases(list(_COMPLEX_SPECIAL_PAYLOADS), quick=[])
)
@pytest.mark.parametrize("dtype", tu.selected_cases(_COMPLEX_DTYPES, quick=[]))
def test_matrix_H_special_values_complex(scenario, dtype):
    inp = torch.tensor(
        _COMPLEX_SPECIAL_PAYLOADS[scenario], dtype=dtype, device=flag_gems.device
    ).reshape(3, 1)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.matrix_H(ref_inp)
    required_device = inp.device
    res_out = flag_gems.matrix_H(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize("shape", _OVERLOAD_SHAPES)
@pytest.mark.parametrize("dtype", _MATRIX_H_DTYPES)
def test_matrix_H_overload_a(shape, dtype):
    # aten::matrix_H.a has the same single-tensor signature as .default, so the
    # candidate is reached through the one public name for both overloads.
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.matrix_H.a(ref_inp)
    required_device = inp.device
    res_out = flag_gems.matrix_H(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize("shape", _INVALID_RANK_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_matrix_H_rejects_invalid_rank(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.matrix_H(inp)


@pytest.mark.matrix_H
@pytest.mark.parametrize("value", [3.14, 7, [1.0, 2.0], "matrix"])
def test_matrix_H_rejects_non_tensor(value):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.matrix_H(value)
