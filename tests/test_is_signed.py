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

"""Correctness tests for ``aten::is_signed``.

``aten::is_signed(Tensor self) -> bool`` is a dtype predicate: it reads no
stored element and returns a Python bool, so dtype is the axis that carries the
semantics and the value-range / broadcast / backward / scalar-operand grids do
not apply.
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
    if dtype == torch.uint64 and list(value_range) == ["0", "max"]:
        # Preserve the full signed-int64 sample and its modulo-2**64 cast.
        device = "cpu" if flag_gems.vendor_name == "metax" else flag_gems.device
        inp = torch.randint(
            torch.iinfo(torch.int64).min,
            torch.iinfo(torch.int64).max,
            shape,
            dtype=torch.int64,
            device=device,
        ).to(torch.uint64)
        if flag_gems.vendor_name == "metax":
            inp = _copy_to(inp, flag_gems.device)
    else:
        cpu_encoding = flag_gems.vendor_name == "ascend" and dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
        cpu_encoding = cpu_encoding or (
            flag_gems.vendor_name == "kunlunxin"
            and (
                dtype in (torch.uint16, torch.uint32)
                or (
                    dtype == torch.uint64
                    and list(value_range) in (["-1", "1"], ["0", "1"])
                )
            )
        )
        if cpu_encoding:
            inp = _copy_to(_cpu_input(dtype, shape, value_range), flag_gems.device)
        else:
            inp = tu.make_input(dtype, shape, value_range)
    return _check_input(inp, dtype, shape)


def _make_special_input(dtype, scenario):
    if flag_gems.vendor_name == "ascend" and dtype in (
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


def _conj_input():
    if flag_gems.vendor_name in ("ascend", "kunlunxin"):
        inp = _copy_to(
            torch.ones(6, dtype=torch.complex64, device="cpu"), flag_gems.device
        )
    else:
        inp = torch.ones(6, dtype=torch.complex64, device=flag_gems.device)
    return _check_input(inp, torch.complex64, (6,)).conj()


def _channels_last_input():
    device = "cpu" if flag_gems.vendor_name == "ascend" else flag_gems.device
    inp = torch.ones((2, 3, 4, 4), dtype=torch.int32, device=device).to(
        memory_format=torch.channels_last
    )
    if flag_gems.vendor_name == "ascend":
        inp = _copy_to(inp, flag_gems.device)
    return _check_input(inp, torch.int32, (2, 3, 4, 4))


def _to_reference(inp):
    if utils.TO_CPU and inp.device.type == "meta":
        # Meta has no payload. is_signed only observes dtype; retain geometry
        # when constructing the independent CPU metadata reference.
        reference = torch.empty_strided(
            inp.shape, inp.stride(), dtype=inp.dtype, device="cpu"
        )
    elif utils.TO_CPU and flag_gems.vendor_name == "ascend" and inp.is_neg():
        # A typed copy resolves lazy negation using an unavailable int16 kernel.
        reference = _copy_to(inp, "cpu")
    else:
        reference = tu.to_reference(inp)
    assert reference.dtype == inp.dtype and reference.shape == inp.shape
    assert reference.layout == inp.layout
    assert reference.device == _device("cpu" if utils.TO_CPU else inp.device)
    return reference


def _check_layout(inp, make_case):
    label = next(name for name, builder in LAYOUT_CASES if builder is make_case)
    specs = {
        "transposed_int32": (torch.int32, (6, 4), (1, 6), 0),
        "storage_offset_int8": (torch.int8, (4, 5), (6, 1), 1),
        "expanded_stride0_uint8": (torch.uint8, (4, 6), (0, 1), 0),
        "channels_last_int32": (torch.int32, (2, 3, 4, 4), (48, 1, 12, 3), 0),
        "conj_view_complex64": (torch.complex64, (6,), (1,), 0),
        "neg_view_int16": (torch.int16, (4, 6), (6, 1), 0),
        "empty_dim_int32": (torch.int32, (0,), (1,), 0),
        "meta_uint8": (torch.uint8, (2, 3), (3, 1), 0),
        "sparse_coo_int32": (torch.int32, (4, 6), None, None),
        "sparse_coo_uint8": (torch.uint8, (4, 6), None, None),
    }
    dtype, shape, stride, offset = specs[label]
    assert inp.dtype == dtype and inp.shape == torch.Size(shape)
    assert inp.device == _device("meta" if label == "meta_uint8" else flag_gems.device)
    if stride is None:
        assert inp.layout == torch.sparse_coo
    else:
        assert inp.layout == torch.strided
        assert inp.stride() == stride and inp.storage_offset() == offset
        assert inp.is_conj() == (label == "conj_view_complex64")
        assert inp.is_neg() == (label == "neg_view_int16")


# Probed against torch.ops.aten.is_signed on the active backend: True for the
# signed integer kinds and for every float/complex kind, False for bool and the
# unsigned integer kinds.
_SIGNED_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex32,
    torch.complex64,
    torch.complex128,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

_UNSIGNED_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
]

# Backend capability flags are the shared accuracy_utils booleans; a dtype with
# no entry is a baseline type the backend always provides.
_DTYPE_CAPABILITY = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.complex128: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}

SUPPORTED_DTYPES = [
    dtype
    for dtype in _SIGNED_DTYPES + _UNSIGNED_DTYPES
    if _DTYPE_CAPABILITY.get(dtype, True)
]

_SUPPORTED_FLOAT_DTYPES = [
    dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point
]


# Exercise shared ranges even though the native predicate only reads dtype.
@pytest.mark.is_signed
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_is_signed_dtype_and_shape(dtype, shape, value_range):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.is_signed(ref_inp)
    res_out = flag_gems.is_signed(inp)

    assert isinstance(res_out, bool), type(res_out)
    assert res_out == ref_out


# Input forms whose strides, storage offset, layout, aliasing or device must not
# change the dtype-derived answer. No broadcast, backward or tensor-vs-scalar
# workload exists: the signature is unary, the result is a Python bool with no
# autograd graph, and a non-tensor argument is rejected (negative cases below).
LAYOUT_CASES = [
    (
        "transposed_int32",
        lambda: torch.ones((4, 6), dtype=torch.int32, device=flag_gems.device).t(),
    ),
    (
        "storage_offset_int8",
        lambda: torch.ones((4, 6), dtype=torch.int8, device=flag_gems.device)[:, 1:],
    ),
    (
        "expanded_stride0_uint8",
        lambda: torch.ones((1, 6), dtype=torch.uint8, device=flag_gems.device).expand(
            4, 6
        ),
    ),
    (
        "channels_last_int32",
        lambda: _channels_last_input(),
    ),
    (
        "conj_view_complex64",
        lambda: _conj_input(),
    ),
    (
        "neg_view_int16",
        lambda: torch._neg_view(
            torch.ones((4, 6), dtype=torch.int16, device=flag_gems.device)
        ),
    ),
    (
        "empty_dim_int32",
        lambda: torch.empty(0, dtype=torch.int32, device=flag_gems.device),
    ),
    (
        "meta_uint8",
        lambda: torch.empty((2, 3), dtype=torch.uint8, device="meta"),
    ),
    (
        "sparse_coo_int32",
        lambda: torch.ones(
            (4, 6), dtype=torch.int32, device=flag_gems.device
        ).to_sparse(),
    ),
    (
        "sparse_coo_uint8",
        lambda: torch.ones(
            (4, 6), dtype=torch.uint8, device=flag_gems.device
        ).to_sparse(),
    ),
]


@pytest.mark.is_signed
@pytest.mark.parametrize(
    "make_case",
    [case for _, case in LAYOUT_CASES],
    ids=[name for name, _ in LAYOUT_CASES],
)
def test_is_signed_input_layout(make_case):
    inp = make_case()
    _check_layout(inp, make_case)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.is_signed(ref_inp)
    res_out = flag_gems.is_signed(inp)

    assert isinstance(res_out, bool), type(res_out)
    assert res_out == ref_out


# Default-only: the reference ignores element contents, so NaN/Inf payloads must
# leave the bool unchanged.
SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(_SUPPORTED_FLOAT_DTYPES), quick=[]
)


@pytest.mark.is_signed
@pytest.mark.parametrize(
    "dtype,scenario",
    SPECIAL_VALUE_CASES,
    ids=[
        f"{str(dtype).removeprefix('torch.')}-{scenario}"
        for dtype, scenario in SPECIAL_VALUE_CASES
    ],
)
def test_is_signed_special_values(dtype, scenario):
    inp = _make_special_input(dtype, scenario)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten.is_signed(ref_inp)
    res_out = flag_gems.is_signed(inp)

    assert isinstance(res_out, bool), type(res_out)
    assert res_out == ref_out


# The schema is aten::is_signed(Tensor self); a non-tensor argument is invalid.
NON_TENSOR_CASES = [
    ("undefined_tensor", None),
    ("int_argument", 1),
    ("float_argument", 1.0),
    ("list_argument", [1, 2]),
    ("string_argument", "x"),
]


@pytest.mark.is_signed
@pytest.mark.parametrize(
    "value",
    [value for _, value in NON_TENSOR_CASES],
    ids=[name for name, _ in NON_TENSOR_CASES],
)
def test_is_signed_rejects_non_tensor(value):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_signed(value)


@pytest.mark.is_signed
def test_is_signed_rejects_extra_argument():
    # The schema takes exactly one argument; the native operator raises
    # "expected at most 1 argument(s) but received 2".
    inp = torch.ones(4, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_signed(inp, inp)


# at::isSignedType has no quantized branch: the native operator raises
# "isSignedType not supported for quantized types". The fixture is the native
# CPU quantized type, so no device quantize kernel is assumed and the candidate
# receives exactly the argument type the reference rejects. quint4x2 is absent
# because torch.quantize_per_tensor cannot construct it on this build.
@pytest.mark.is_signed
@pytest.mark.parametrize(
    "qscheme_dtype", [torch.quint8, torch.qint8, torch.qint32], ids=str
)
def test_is_signed_rejects_quantized(qscheme_dtype):
    inp = torch.quantize_per_tensor(torch.ones(4, device="cpu"), 0.1, 0, qscheme_dtype)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_signed(inp)
