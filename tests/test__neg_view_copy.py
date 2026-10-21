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

# Select every overload and contract case with the operator marker.
pytestmark = pytest.mark.underscore_neg_view_copy

# Materialize negated values in independent contiguous storage.
_NEG_VIEW_COPY_DTYPES = (
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
)

_UNSUPPORTED_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2, torch.bool]

_NEG_VIEW_COPY_SHAPES = [(17,), (12, 13)] + tu.selected_shapes()

_NEG_VIEW_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]

_NEG_VIEW_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]

_NEG_VIEW_COPY_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]

# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _NEG_VIEW_COPY_DTYPES
    for value_range in tu.selected_ranges()
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


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
    if not utils.TO_CPU and flag_gems.vendor_name != "kunlunxin":
        return tu.to_reference(inp)
    reference = _cpu_storage_view(inp)
    if inp.is_conj():
        reference = reference.conj()
    if inp.is_neg():
        reference = torch._neg_view(reference)
    return reference.requires_grad_(inp.requires_grad)


class _PhysicalDeviceCopy(torch.autograd.Function):
    """Logical device copy whose backward retains the native CPU autograd chain.

    Kunlunxin typed copies can drop lazy negative bits. Transfer physical bytes
    and resolve logical flags on CPU, keeping the identity derivative explicit.
    """

    @staticmethod
    def forward(ctx, inp, device):
        ctx.source_device = inp.device
        cpu = _cpu_storage_view(inp)
        if inp.is_conj():
            cpu = cpu.conj()
        if inp.is_neg():
            cpu = torch._neg_view(cpu)
        return cpu.resolve_conj().resolve_neg().to(device)

    @staticmethod
    def backward(ctx, grad):
        return _PhysicalDeviceCopy.apply(grad, ctx.source_device), None


def cpu_oracle(inp, *, out=None):
    """Trusted CPU native operation, adapted to the caller's required device.

    Used to bootstrap ref-as-candidate on Kunlunxin: its native copy returns
    unnegated physical values. This oracle does not validate a generated kernel.
    """
    if not isinstance(inp, torch.Tensor):
        return torch.ops.aten._neg_view_copy(inp)
    cpu_inp = (
        inp
        if inp.device.type == "cpu"
        else _PhysicalDeviceCopy.apply(inp, torch.device("cpu"))
    )
    if out is None:
        result = torch.ops.aten._neg_view_copy(cpu_inp)
        return (
            result
            if inp.device.type == "cpu"
            else _PhysicalDeviceCopy.apply(result, inp.device)
        )
    if inp.device != out.device:
        raise RuntimeError("input and out must use the same device")
    if inp.device.type == "cpu":
        return torch.ops.aten._neg_view_copy.out(inp, out=out)
    cpu_out = _PhysicalDeviceCopy.apply(out, torch.device("cpu"))
    torch.ops.aten._neg_view_copy.out(cpu_inp, out=cpu_out)
    if out.shape != cpu_out.shape:
        out.resize_(cpu_out.shape)
    out.copy_(cpu_out.to(out.device))
    return out


def _reference(inp, *, out=None):
    if flag_gems.vendor_name == "kunlunxin":
        return cpu_oracle(inp, out=out)
    if (
        isinstance(inp, torch.Tensor)
        and flag_gems.vendor_name == "ascend"
        and inp.dtype in (torch.uint8, torch.int16)
        and inp.numel()
    ):
        pytest.skip(
            "Ascend native negative copy lacks nonempty uint8/int16 Neg kernels"
        )
    if out is None:
        return torch.ops.aten._neg_view_copy(inp)
    return torch.ops.aten._neg_view_copy.out(inp, out=out)


def _candidate(inp, *, out=None):
    if not isinstance(inp, torch.Tensor):
        return flag_gems._neg_view_copy(inp)
    storage = inp.untyped_storage()
    before = _cpu_storage_view(inp)
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
    out_storage = out.untyped_storage() if out is not None else None
    out_geometry = (
        (
            out.device,
            out.dtype,
            out.shape,
            out.stride(),
            out.storage_offset(),
            out.is_neg(),
            out.is_conj(),
        )
        if out is not None
        else None
    )
    result = (
        flag_gems._neg_view_copy(inp)
        if out is None
        else flag_gems._neg_view_copy(inp, out=out)
    )
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

    def bytes_of(t):
        return torch.empty(0, dtype=torch.uint8).set_(
            t.untyped_storage(), 0, (t.untyped_storage().nbytes(),), (1,)
        )

    torch.testing.assert_close(bytes_of(after), bytes_of(before), rtol=0, atol=0)
    if out is not None:
        assert result is out
        assert out.untyped_storage()._cdata == out_storage._cdata
        assert out.untyped_storage().data_ptr() == out_storage.data_ptr()
        assert (
            out.device,
            out.dtype,
            out.shape,
            out.stride(),
            out.storage_offset(),
            out.is_neg(),
            out.is_conj(),
        ) == out_geometry
    return result


def _assert_values_equal(result, reference):
    assert result.device == _expected_device()
    assert reference.device == (
        torch.device("cpu")
        if utils.TO_CPU or flag_gems.vendor_name == "kunlunxin"
        else result.device
    )
    assert result.dtype == reference.dtype and result.shape == reference.shape
    assert result.layout == reference.layout
    torch.testing.assert_close(
        _numeric_values(result),
        _numeric_values(reference),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp):
    assert res_out is not inp
    assert not torch._C._is_alias_of(res_out, inp)
    assert res_out.is_contiguous()
    assert not res_out.is_neg()
    # Zero-element tensors carry a null data pointer on every tensor, so the
    # no-alias check is only meaningful for non-empty inputs.
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    _assert_values_equal(inp, ref_inp)
    _assert_values_equal(res_out, ref_out)


@pytest.mark.underscore_neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy(shape, dtype):
    inp = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    # to_reference creates an independent snapshot for the input mutation check.
    ref_inp = _to_reference(inp)

    ref_out = _reference(ref_inp)
    res_out = _candidate(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_neg_view_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_copy_value_ranges(shape, dtype, value_range):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _to_reference(inp)

    ref_out = _reference(ref_inp)
    res_out = _candidate(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.neg_view_copy_out
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_out(shape, dtype):
    inp = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_inp = _to_reference(inp)

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    _reference(ref_inp, out=ref_out)
    res_ret = _candidate(inp, out=out)

    # The .out variant must write into and return the caller's buffer itself.
    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_out, inp, ref_inp)


@pytest.mark.neg_view_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(("dtype", "value_range"), _RANGE_CASES)
def test__neg_view_copy_out_value_ranges(shape, dtype, value_range):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _to_reference(inp)

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    _reference(ref_inp, out=ref_out)
    res_ret = _candidate(inp, out=out)

    assert res_ret is out
    _assert_copy_semantics(res_ret, ref_out, inp, ref_inp)


@pytest.mark.underscore_neg_view_copy
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
def test__neg_view_copy_special_values(dtype, scenario):
    values = _make_special_input(dtype, scenario)
    ref_inp = _to_reference(values)

    ref_out = _reference(ref_inp)
    res_out = _candidate(values)

    _assert_copy_semantics(res_out, ref_out, values, ref_inp)
    # Exact numerical equality does not distinguish the signs of zero.
    zeros = _numeric_values(values) == 0
    torch.testing.assert_close(
        torch.signbit(_numeric_values(res_out)[zeros]),
        torch.signbit(_numeric_values(ref_out)[_numeric_values(ref_inp) == 0]),
        rtol=0,
        atol=0,
    )


@pytest.mark.underscore_neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_non_contiguous(shape, dtype):
    base = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_base = _to_reference(base)
    inp = base.transpose(-1, -2)
    ref_inp = ref_base.transpose(-1, -2)
    assert not inp.is_contiguous()

    ref_out = _reference(ref_inp)
    res_out = _candidate(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _NEG_VIEW_COPY_DTYPES)
def test__neg_view_copy_empty(shape, dtype):
    inp = _make_input(
        dtype, shape, ["0", "max"] if dtype == torch.uint8 else ["-1", "1"]
    )
    ref_inp = _to_reference(inp)

    ref_out = _reference(ref_inp)
    res_out = _candidate(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.underscore_neg_view_copy
@pytest.mark.parametrize("shape", _NEG_VIEW_COPY_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test__neg_view_copy_backward(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = _to_reference(inp)
    ref_grad = _to_reference(grad)

    ref_out = _reference(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    res_out = _candidate(inp)
    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    assert res_in_grad.device == inp.device and res_in_grad.dtype == inp.dtype
    assert res_in_grad.shape == ref_in_grad.shape
    torch.testing.assert_close(
        _numeric_values(res_in_grad),
        _numeric_values(ref_in_grad),
        rtol=flag_gems.testing.RESOLUTION[dtype],
        atol=1e-4,
        equal_nan=True,
    )


@pytest.mark.underscore_neg_view_copy
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test__neg_view_copy_rejects_unsupported_dtypes(dtype):
    inp = torch.zeros(4, dtype=dtype, device="cpu").to(flag_gems.device)
    with pytest.raises(RuntimeError):
        _reference(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _candidate(inp)


@pytest.mark.underscore_neg_view_copy
def test__neg_view_copy_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        _reference(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _candidate(3.14)
