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

# Toggle the lazy neg bit while preserving input storage and layout.
# FP8/bool views exist, but their negated values cannot be materialized.
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
_UNMATERIALIZABLE_DTYPES = _FP8_DTYPES + [torch.bool]

_VALUE_DTYPES = (
    [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
    ]
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.complex64]
)

_ALL_TEST_DTYPES = _VALUE_DTYPES + _UNMATERIALIZABLE_DTYPES

_NEG_VIEW_SHAPES = [(17,), (12, 13), (0,), (3, 0), (2, 0, 4)] + tu.selected_shapes()

_NEG_VIEW_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]

_NEG_VIEW_TOGGLE_SHAPES = [(16, 32), (4, 8, 16)]

_NEG_VIEW_MUTATION_SHAPES = [(16, 32), (4, 8, 16)]

_NEG_VIEW_BACKWARD_SHAPES = [(16, 64), (7, 13, 29), (0,), (3, 0)]

# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _ALL_TEST_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


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
    # Decode FP8 before resolving lazy bits: CPU FP8 arithmetic is limited.
    if result.dtype in _FP8_DTYPES:
        result = result.float()
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    return result.resolve_conj().resolve_neg()


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


def _to_reference(inp):
    if not utils.TO_CPU:
        return tu.to_reference(inp)
    reference = _cpu_storage_view(inp)
    if inp.is_conj():
        reference = reference.conj()
    if inp.is_neg():
        reference = torch._neg_view(reference)
    return reference.requires_grad_(inp.requires_grad)


def _candidate(inp):
    # Capture the caller's storage before invocation, including unused view holes.
    before = _cpu_storage_view(inp)
    storage = inp.untyped_storage()
    raw_before = (
        torch.empty(0, dtype=torch.uint8, device="cpu")
        .set_(before.untyped_storage(), 0, (before.untyped_storage().nbytes(),), (1,))
        .clone()
    )
    geometry = (
        inp.device,
        inp.dtype,
        inp.shape,
        inp.stride(),
        inp.storage_offset(),
        inp.is_neg(),
        inp.is_conj(),
        inp.requires_grad,
    )
    result = flag_gems._neg_view(inp)
    assert inp.untyped_storage()._cdata == storage._cdata
    assert (
        inp.device,
        inp.dtype,
        inp.shape,
        inp.stride(),
        inp.storage_offset(),
        inp.is_neg(),
        inp.is_conj(),
        inp.requires_grad,
    ) == geometry
    after = _cpu_storage_view(inp)
    raw_after = torch.empty(0, dtype=torch.uint8, device="cpu").set_(
        after.untyped_storage(), 0, (after.untyped_storage().nbytes(),), (1,)
    )
    torch.testing.assert_close(raw_after, raw_before, rtol=0, atol=0)
    assert result is not inp
    return result


def _assert_values_equal(res_out, ref_out):
    # For FP8/bool, clear the neg bit on aliases to inspect unchanged storage.
    if ref_out.dtype in _UNMATERIALIZABLE_DTYPES and ref_out.is_neg():
        res_out = torch.ops.aten._neg_view(res_out)
        ref_out = torch.ops.aten._neg_view(ref_out)
    assert res_out.device == _expected_device()
    assert ref_out.device == (torch.device("cpu") if utils.TO_CPU else res_out.device)
    assert res_out.dtype == ref_out.dtype and res_out.shape == ref_out.shape
    torch.testing.assert_close(
        _numeric_values(res_out),
        _numeric_values(ref_out),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def _assert_view_semantics(res_out, ref_out, inp):
    assert res_out.device == inp.device
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.is_neg() == ref_out.is_neg()
    # Empty tensors can have equal null pointers without sharing storage.
    assert torch._C._is_alias_of(res_out, inp)


@pytest.mark.underscore_neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_SHAPES)
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__neg_view(shape, dtype):
    inp = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _candidate(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)
    assert res_out.is_neg()


@pytest.mark.underscore_neg_view
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_value_ranges(shape, dtype, value_range):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _candidate(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)


@pytest.mark.underscore_neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_SHAPES)
@pytest.mark.parametrize("dtype", _UNMATERIALIZABLE_DTYPES)
def test__neg_view_unmaterializable_dtypes(shape, dtype):
    inp = _make_input(dtype, shape, ["0", "1"])
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _candidate(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    assert res_out.is_neg()
    _assert_values_equal(res_out, ref_out)


@pytest.mark.underscore_neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _ALL_TEST_DTYPES)
def test__neg_view_non_contiguous(shape, dtype):
    base = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_base = _to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _candidate(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)


@pytest.mark.underscore_neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_TOGGLE_SHAPES)
@pytest.mark.parametrize("dtype", _ALL_TEST_DTYPES)
def test__neg_view_toggle(shape, dtype):
    base = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_base = _to_reference(base)

    inp = torch.ops.aten._neg_view(base)
    ref_inp = torch.ops.aten._neg_view(ref_base)
    assert inp.is_neg()

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _candidate(inp)

    _assert_view_semantics(res_out, ref_out, base)
    _assert_values_equal(res_out, ref_out)
    assert not res_out.is_neg()


@pytest.mark.underscore_neg_view
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
def test__neg_view_special_values(dtype, scenario):
    values = _make_special_input(dtype, scenario)
    ref_inp = _to_reference(values)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _candidate(values)

    _assert_view_semantics(res_out, ref_out, values)
    _assert_values_equal(res_out, ref_out)
    # Exact numerical equality does not distinguish the signs of zero.
    # NPU boolean indexing/signbit can erase signed zero. Observe physical
    # storage only after checking the original candidate's device and view ABI.
    zeros = _numeric_values(values) == 0
    torch.testing.assert_close(
        torch.signbit(_numeric_values(res_out)[zeros]),
        torch.signbit(_numeric_values(ref_out)[_numeric_values(ref_inp) == 0]),
        rtol=0,
        atol=0,
    )


@pytest.mark.underscore_neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_MUTATION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__neg_view_mutation(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        pytest.skip("Kunlunxin fill_ through a negative view ignores the negative bit")
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = _to_reference(inp)

    res_out = _candidate(inp)
    ref_out = torch.ops.aten._neg_view(ref_inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)

    res_out.fill_(2.5)
    ref_out.fill_(2.5)

    _assert_values_equal(res_out, ref_out)
    # fill_ through a neg view writes -2.5 into the base storage, so the input
    # (no neg bit) materializes to -2.5 on both sides.
    _assert_values_equal(inp, ref_inp)


@pytest.mark.underscore_neg_view
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_FP8_DTYPES))
)
def test__neg_view_fp8_special_values(dtype, scenario):
    inp = _make_special_input(dtype, scenario)
    ref_inp = _to_reference(inp)
    ref_out = torch.ops.aten._neg_view(ref_inp)
    res_out = _candidate(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)
    # The view must preserve storage bytes, including signed zeros and NaNs.
    _assert_values_equal(inp.view(torch.uint8), ref_inp.view(torch.uint8))


@pytest.mark.underscore_neg_view
@pytest.mark.parametrize("shape", _NEG_VIEW_BACKWARD_SHAPES)
# FP8 backward calls aten.neg, which has no CPU/CUDA FP8 kernel.
@pytest.mark.parametrize(
    "dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES + [torch.complex64])
)
def test__neg_view_backward(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.complex64:
        pytest.skip(
            "Kunlunxin native negative-view complex64 backward is not implemented"
        )
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = _to_reference(inp)
    ref_grad = _to_reference(grad)

    ref_out = torch.ops.aten._neg_view(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = _candidate(inp)
    _assert_view_semantics(res_out, ref_out, inp)
    _assert_values_equal(res_out, ref_out)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    _assert_values_equal(res_in_grad, ref_in_grad)


@pytest.mark.underscore_neg_view
def test__neg_view_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._neg_view(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._neg_view(3.14)
