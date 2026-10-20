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


def _assert_snapshot_equal(result, reference, expected_device):
    # Validate the original operand ABI before observing values on CPU.
    assert result.device == expected_device
    assert reference.device == _device("cpu" if utils.TO_CPU else expected_device)
    assert result.dtype == reference.dtype and result.shape == reference.shape
    assert result.layout == reference.layout
    assert (result.is_conj(), result.is_neg()) == (
        reference.is_conj(),
        reference.is_neg(),
    )
    torch.testing.assert_close(
        _copy_to(result, "cpu"),
        _copy_to(reference, "cpu"),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


# aten::is_floating_point(Tensor self) -> bool reads self.dtype only, so the
# answer must not depend on values, shape, strides, storage offset or lazy
# conj/neg state. Broadcast and backward do not apply (one operand, Python bool
# result) and the operator has no .out schema (probed absent natively).
_IS_FLOATING_POINT_DTYPES = (
    list(tu.REQUIRED_DTYPES)
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.bool, torch.complex64, torch.complex32]
)


def _layout_view(base, layout):
    """Build the operand of one probed layout; it keeps sharing base storage."""
    if layout == "transposed":
        return base.transpose(0, 1)
    if layout == "column_slice":
        return base[:, ::2]
    if layout == "window":
        return base[2:6, 1:5]
    if layout == "expanded":
        return base[:1].expand(3, base.shape[1])
    if layout == "neg_view":
        return torch._neg_view(base)
    if layout == "conj":
        return base.conj()
    raise AssertionError(f"unknown layout {layout}")


# Small stride, offset and lazy-flag cases stay in quick mode.
_LAYOUT_CASES = [
    ((4, 6), "transposed", torch.float32),
    ((8, 12), "column_slice", torch.float16),
    ((10, 8), "window", torch.int8),
    ((1, 8), "expanded", torch.bfloat16),
    ((4, 6), "neg_view", torch.float32),
    ((4, 6), "conj", torch.complex64),
]

_EMPTY_CASES = [
    (shape, dtype)
    for shape in [(0,), (0, 4), (2, 0, 3)]
    for dtype in (torch.float32, torch.int8)
]

# One row per representable nan/inf scenario of each floating dtype: e4m3fn has
# no inf encoding and therefore contributes only its nan row.
_SPECIAL_VALUE_CASES = tu.selected_cases(
    tu.special_value_cases(
        [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
        + ([torch.float64] if utils.fp64_is_supported else [])
    ),
    quick=[],
)

_NON_TENSOR_CASES = [
    pytest.param(3.14, id="python_float"),
    pytest.param([1.0, 2.0], id="list"),
    pytest.param("float32", id="string"),
]


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _IS_FLOATING_POINT_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.is_floating_point
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test_is_floating_point(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.is_floating_point(ref_inp)
    res_out = flag_gems.is_floating_point(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_floating_point
@pytest.mark.parametrize("shape,layout,dtype", _LAYOUT_CASES)
def test_is_floating_point_layout(shape, layout, dtype):
    base = _make_input(dtype, shape, ["-1", "1"])
    inp = _layout_view(base, layout)
    # The snapshots keep strides, offset and the lazy conj/neg bits, so they catch
    # a candidate that rewrote or materialised the operand it was handed.
    base_snapshot = _reference(base)
    view_snapshot = _reference(inp.detach())
    ref_inp = _layout_view(base_snapshot, layout)
    expected_device = inp.device

    ref_out = torch.ops.aten.is_floating_point(ref_inp)
    res_out = flag_gems.is_floating_point(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out
    assert inp.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    assert (inp.stride(), inp.storage_offset()) == (
        view_snapshot.stride(),
        view_snapshot.storage_offset(),
    )
    _assert_snapshot_equal(inp, view_snapshot, expected_device)
    _assert_snapshot_equal(base, base_snapshot, expected_device)


@pytest.mark.is_floating_point
@pytest.mark.parametrize("shape,dtype", _EMPTY_CASES)
def test_is_floating_point_empty_operand(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])

    ref_out = torch.ops.aten.is_floating_point(_reference(inp))
    res_out = flag_gems.is_floating_point(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_floating_point
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_VALUE_CASES)
def test_is_floating_point_special_values(dtype, scenario):
    inp = _make_special_input(dtype, scenario)

    ref_out = torch.ops.aten.is_floating_point(_reference(inp))
    res_out = flag_gems.is_floating_point(inp)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_floating_point
def test_is_floating_point_optional_none():
    # The schema's optional-Tensor form accepts None and reports it as not
    # floating, so this is a valid workload, not an invalid input.
    ref_out = torch.ops.aten.is_floating_point(None)
    res_out = flag_gems.is_floating_point(None)

    assert isinstance(res_out, bool)
    assert res_out == ref_out


@pytest.mark.is_floating_point
@pytest.mark.parametrize("bad", _NON_TENSOR_CASES)
def test_is_floating_point_rejects_non_tensor(bad):
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_floating_point(bad)


@pytest.mark.is_floating_point
def test_is_floating_point_requires_argument():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_floating_point()
