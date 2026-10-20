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

"""Correctness tests for aten::is_neg.

is_neg reads the lazy neg bit recorded by torch._neg_view; the answer is a
Python bool independent of values, rank, layout and dtype.  Setting and reading
the bit is metadata only, so no test materializes a flagged tensor (resolving a
neg view of bool or FP8 has no neg kernel).  The operator has one operand and
its result is not differentiable, so broadcast, tensor/scalar and backward
dimensions do not apply.
"""

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


def _reference(inp):
    return _copy_to(inp, "cpu" if utils.TO_CPU else inp.device)


def _check_input(inp, dtype, shape):
    assert inp.dtype == dtype and inp.shape == torch.Size(shape)
    assert inp.device == _device(flag_gems.device)
    return inp


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        low, high = (tu.resolve_bound(v, dtype) for v in value_range)
        if low == high:
            value = torch.full(shape, low, dtype=torch.float32, device="cpu")
        else:
            value = torch.testing.make_tensor(
                shape, dtype=torch.float32, device="cpu", low=low, high=high
            )
        inp = _copy_to(value.to(dtype), flag_gems.device)
    else:
        inp = tu.make_input(dtype, shape, value_range)
    return _check_input(inp, dtype, shape)


def _make_special_input(dtype, scenario):
    cpu_encoding = flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if cpu_encoding:
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


def _assert_payload_equal(result, reference):
    # Validate the selected reference ABI before observing values on CPU.
    assert result.dtype == reference.dtype and result.shape == reference.shape
    assert result.layout == reference.layout
    expected = _device("cpu" if utils.TO_CPU else flag_gems.device)
    assert result.device == reference.device == expected
    torch.testing.assert_close(
        _copy_to(result, "cpu"),
        _copy_to(reference, "cpu"),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def _input_marks(dtype):
    return ()


_FP64_SUPPORTED = utils.fp64_is_supported

# int8/uint8/FP8/bool carry the flag like any other dtype, so only the dtypes
# that need backend support for the tensor itself are gated.
_DTYPES = (
    tu.REQUIRED_DTYPES
    + [torch.bool, torch.complex64]
    + ([torch.float64, torch.complex128] if _FP64_SUPPORTED else [])
)


_LAYOUT_SHAPES = {
    "base": (4, 3),
    "slice": (5, 3),
    "transpose": (4, 3),
    "expand": (5, 1),
    "narrow": (4, 3),
    "scalar": (),
    "empty": (0, 3),
}


def _layout_view(tensor, layout):
    if layout == "slice":
        return tensor[1:3]
    if layout == "transpose":
        return tensor.transpose(0, 1)
    if layout == "expand":
        return tensor.expand(5, 4)
    if layout == "narrow":
        return tensor.narrow(1, 1, 2)
    return tensor


def _with_flags(tensor, conj, neg_count, conj_first):
    if conj and conj_first:
        tensor = tensor.conj()
    for _ in range(neg_count):
        tensor = torch._neg_view(tensor)
    if conj and not conj_first:
        tensor = tensor.conj()
    return tensor


def _flag_form_pair(dtype, layout, conj, neg_count, conj_first):
    # The reference operand is built independently and gets the same layout and
    # flags, so no flagged tensor ever has to cross devices.
    plain = _make_input(dtype, _LAYOUT_SHAPES[layout], ["-1", "1"])
    inp = _with_flags(_layout_view(plain, layout), conj, neg_count, conj_first)
    ref_inp = _with_flags(
        _layout_view(_reference(plain), layout), conj, neg_count, conj_first
    )
    return inp, ref_inp


# One row per flag/layout composition. conj() only sets the conjugate bit on
# complex dtypes, so the conj rows carry complex64 (and complex128 when the
# backend supports fp64).  These rows are small, so quick keeps all of them.
_FLAG_FORMS = [
    (torch.float32, "base", False, 0, True),
    (torch.float32, "base", False, 1, True),
    (torch.float32, "base", False, 2, True),
    (torch.float32, "slice", False, 1, True),
    (torch.float32, "transpose", False, 1, True),
    (torch.float32, "expand", False, 1, True),
    (torch.float32, "narrow", False, 1, True),
    (torch.float32, "scalar", False, 1, True),
    (torch.float32, "empty", False, 1, True),
    (torch.int8, "empty", False, 1, True),
    (torch.complex64, "base", True, 0, True),
    (torch.complex64, "base", True, 1, True),
    (torch.complex64, "base", True, 1, False),
] + ([(torch.complex128, "base", True, 1, True)] if _FP64_SUPPORTED else [])

# Read-only contract: the flag read must not change stride, storage offset,
# storage identity, lazy flags, grad state or payload.
_STATE_ROWS = [
    (torch.bool, (3, 4), False, False),
    (torch.int64, (5,), False, False),
    (torch.float32, (2, 3), True, False),
    (torch.complex64, (2, 3), True, False),
    (torch.float32, (2, 3), False, True),
]

_SPECIAL_FLOAT_DTYPES = [
    dtype
    for dtype in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if dtype != torch.float64 or _FP64_SUPPORTED
]

_SPECIAL_COMPLEX_DTYPES = [torch.complex64] + (
    [torch.complex128] if _FP64_SUPPORTED else []
)

# Positive special values are default-only.  e4m3fn contributes a nan-only row
# because it cannot represent infinity (tu.special_value_cases).
_SPECIAL_ROWS = tu.selected_cases(
    [
        (dtype, scenario, has_flag)
        for dtype, scenario in tu.special_value_cases(_SPECIAL_FLOAT_DTYPES)
        for has_flag in (False, True)
    ]
    + [
        (dtype, scenario, has_flag)
        for dtype in _SPECIAL_COMPLEX_DTYPES
        for scenario in ("nan", "inf", "mixed")
        for has_flag in (False, True)
    ],
    quick=[],
)

_NEGATIVE_ARGS = [1, 1.5, "is_neg", [1, 2]]


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


@pytest.mark.is_neg
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,value_range",
    [
        pytest.param(dtype, value_range, marks=_input_marks(dtype))
        for dtype, value_range in _DTYPE_RANGE_CASES
    ],
)
def test_is_neg_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.is_neg(ref_inp)
    res_out = flag_gems.is_neg(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_neg
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(dtype, marks=_input_marks(dtype)) for dtype in _DTYPES],
)
def test_is_neg_lazy_neg_view(shape, dtype):
    plain = _make_input(dtype, shape, ["-1", "1"])
    inp = torch._neg_view(plain)
    ref_inp = torch._neg_view(_reference(plain))

    ref_out = torch.ops.aten.is_neg(ref_inp)
    res_out = flag_gems.is_neg(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_neg
@pytest.mark.parametrize(
    "dtype,layout,conj,neg_count,conj_first",
    [
        pytest.param(
            dtype,
            layout,
            conj,
            neg_count,
            conj_first,
            marks=_input_marks(dtype),
        )
        for dtype, layout, conj, neg_count, conj_first in _FLAG_FORMS
    ],
)
def test_is_neg_flag_forms(dtype, layout, conj, neg_count, conj_first):
    inp, ref_inp = _flag_form_pair(dtype, layout, conj, neg_count, conj_first)

    ref_out = torch.ops.aten.is_neg(ref_inp)
    res_out = flag_gems.is_neg(inp)

    # The bit is a parity and is independent of the conjugate bit.
    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_neg
@pytest.mark.parametrize(
    "dtype,shape,has_flag,requires_grad",
    [
        pytest.param(
            dtype,
            shape,
            has_flag,
            requires_grad,
            marks=_input_marks(dtype),
        )
        for dtype, shape, has_flag, requires_grad in _STATE_ROWS
    ],
)
def test_is_neg_keeps_input_state(dtype, shape, has_flag, requires_grad):
    plain = _make_input(dtype, shape, ["-1", "1"])
    if requires_grad:
        plain.requires_grad_(True)
    ref_plain = _reference(plain)
    inp = torch._neg_view(plain) if has_flag else plain
    ref_inp = torch._neg_view(ref_plain) if has_flag else ref_plain
    plain_before = _reference(plain.detach())
    stride, offset = inp.stride(), inp.storage_offset()
    storage = inp.untyped_storage().data_ptr()
    conj = inp.is_conj()

    ref_out = torch.ops.aten.is_neg(ref_inp)
    res_out = flag_gems.is_neg(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out
    assert inp.stride() == stride
    assert inp.storage_offset() == offset
    assert inp.untyped_storage().data_ptr() == storage
    assert inp.is_conj() == conj
    assert inp.is_neg() == has_flag
    assert inp.requires_grad == requires_grad
    # The payload snapshot comes from the unflagged tensor, so comparing it does
    # not have to materialize the negation view.
    _check_input(plain, dtype, shape)
    _assert_payload_equal(_reference(plain), plain_before)


@pytest.mark.is_neg
@pytest.mark.parametrize(
    "dtype,scenario,has_flag",
    [
        pytest.param(
            dtype,
            scenario,
            has_flag,
            marks=_input_marks(dtype),
        )
        for dtype, scenario, has_flag in _SPECIAL_ROWS
    ],
)
def test_is_neg_special_values(dtype, scenario, has_flag):
    plain = _make_special_input(dtype, scenario)
    ref_plain = _reference(plain)
    inp = torch._neg_view(plain) if has_flag else plain
    ref_inp = torch._neg_view(ref_plain) if has_flag else ref_plain

    ref_out = torch.ops.aten.is_neg(ref_inp)
    res_out = flag_gems.is_neg(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_neg
def test_is_neg_undefined_tensor():
    # An undefined tensor is a valid `self` and is not negative.
    ref_out = torch.ops.aten.is_neg(None)
    res_out = flag_gems.is_neg(None)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_neg
@pytest.mark.parametrize("bad_arg", _NEGATIVE_ARGS)
def test_is_neg_rejects_non_tensor(bad_arg):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.is_neg(bad_arg)


@pytest.mark.is_neg
def test_is_neg_requires_input():
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.is_neg()
