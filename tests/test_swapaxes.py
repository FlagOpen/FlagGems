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


def _numeric_values(inp):
    # Copy bytes before decoding: device FP8 casts and complex comparison may
    # be unavailable. Rebuild the logical view and its lazy flags on CPU.
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
    # Decode before resolving lazy negative bits; CPU FP8 neg is unavailable.
    if result.dtype in _FP8_DTYPES:
        result = result.float()
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    return result.resolve_conj().resolve_neg()


def _assert_result_equal(result, reference, *, required_device):
    # Check the original ABI before observing values on CPU. Layout and alias
    # assertions at each call site still inspect the original device tensors.
    assert result.device == required_device
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else result.device)
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape and result.layout == reference.layout
    if (
        result.dtype in _FP8_DTYPES
        or (flag_gems.vendor_name in ("ascend", "kunlunxin") and result.is_complex())
        or (flag_gems.vendor_name == "ascend" and result.dtype == torch.int64)
        # Kunlunxin typed transfer/comparison loses the lazy negative flag.
        or (
            flag_gems.vendor_name == "kunlunxin"
            and (result.is_neg() or reference.is_neg())
        )
    ):
        result, reference = _numeric_values(result), _numeric_values(reference)
    tu.assert_result_equal(result, reference)


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name == "ascend" and dtype in _FP8_DTYPES:
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


def _zeros_like(inp):
    if flag_gems.vendor_name == "ascend" and inp.dtype in _FP8_DTYPES:
        # Keep the write through the returned view; only encode its zero source
        # on CPU because the NPU FP8 zero-fill kernel is unavailable.
        return torch.zeros_like(inp, device="cpu").to(inp.device)
    return torch.zeros_like(inp)


def _make_special_input(dtype, scenario):
    if flag_gems.vendor_name == "ascend" and dtype in _FP8_DTYPES:
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

# The native op accepts every required dtype plus bool/complex (float64 where
# the device supports it); both modes retain all supported dtypes.
_EXTRA_DTYPES = [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _EXTRA_DTYPES.append(torch.float64)
SUPPORTED_DTYPES = tu.REQUIRED_DTYPES + _EXTRA_DTYPES
SUPPORTED_DTYPES = [
    dtype for dtype in SUPPORTED_DTYPES if _DTYPE_FLAGS.get(dtype, True)
]


def _assert_view_matches(res_out, ref_out, inp, ref_inp):
    assert res_out is not inp
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    # swapaxes is a zero-copy view (Tensor(a)): it permutes the size and stride
    # of the two axes, keeps the storage offset and still aliases the input.
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.data_ptr() == inp.data_ptr()
    assert res_out.device == inp.device


# (shape, axis0, axis1) rows: every accepted rank (0-D .. 5-D), the -ndim and
# ndim-1 boundaries, negative axes and equal axes. A 0-D or 1-D tensor only
# accepts the identity swap; any other index is out of range.
_AXIS_ROWS = [
    ((0, 3), 0, 1),
    ((2, 0, 4), 0, 2),
    ((), 0, 0),
    ((1,), 0, 0),
    ((256,), 0, -1),
    ((1024, 1024), 0, 1),
    ((20, 320, 15), 0, 2),
    ((16, 128, 64, 60), 1, 3),
    ((16, 7, 57, 32, 29), 0, 4),
    ((1024, 1024), -1, -2),
    ((20, 320, 15), -3, 1),
    ((16, 128, 64, 60), 3, 0),
    ((16, 128, 64, 60), -4, 2),
    ((20, 320, 15), 2, 1),
    ((16, 7, 57, 32, 29), 1, 1),
]

# Quick keeps the rank boundaries and the first/last, negative and equal-axis
# combinations on the quick shape; only the large-shape combinations are
# trimmed.
_QUICK_AXIS_ROWS = [
    ((0, 3), 0, 1),
    ((2, 0, 4), 0, 2),
    ((), 0, 0),
    ((1,), 0, 0),
    ((2, 19, 7), 0, 2),
    ((2, 19, 7), -3, -1),
    ((2, 19, 7), 1, 1),
]


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in SUPPORTED_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1", tu.selected_cases(_AXIS_ROWS, quick=_QUICK_AXIS_ROWS)
)
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test_swapaxes(shape, axis0, axis1, value_range, dtype):
    if flag_gems.vendor_name == "kunlunxin" and not shape:
        pytest.skip(
            "Kunlunxin native scalar swapaxes copies storage instead of aliasing it"
        )
    required_device = _expected_device()
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes(inp, axis0, axis1)

    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    _assert_result_equal(res_out, ref_out, required_device=required_device)


# Sliced inputs: non-contiguous strides and a non-zero storage offset, which the
# returned view must keep. Both rows are small, so quick keeps them.
_NON_CONTIGUOUS_ROWS = [
    pytest.param(
        (16, 64, 32),
        (slice(None, None, 2), slice(1, None), slice(None, None, 2)),
        0,
        2,
        id="strided",
    ),
    pytest.param(
        (8, 16, 32, 64),
        (slice(None), slice(3, 11), slice(None), slice(0, 30, 3)),
        1,
        3,
        id="offset",
    ),
]


@pytest.mark.swapaxes
@pytest.mark.parametrize("shape,index,axis0,axis1", _NON_CONTIGUOUS_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_swapaxes_non_contiguous(shape, index, axis0, axis1, dtype):
    required_device = _expected_device()
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[index]
    ref_inp = ref_base[index]

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes(inp, axis0, axis1)

    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    _assert_result_equal(res_out, ref_out, required_device=required_device)


# Both rows are small, so quick keeps them.
_MUTATION_ROWS = [((4, 8, 16), 0, 2), ((16, 32, 8, 4), 1, 3)]


@pytest.mark.swapaxes
@pytest.mark.parametrize("shape,axis0,axis1", _MUTATION_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_swapaxes_mutation(shape, axis0, axis1, dtype):
    required_device = _expected_device()
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes(inp, axis0, axis1)
    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    _assert_result_equal(res_out, ref_out, required_device=required_device)

    # Writing through the returned view must reach the original input storage.
    res_out.copy_(_zeros_like(res_out))
    ref_out.copy_(_zeros_like(ref_out))

    _assert_result_equal(inp, ref_inp, required_device=required_device)
    _assert_result_equal(res_out, ref_out, required_device=required_device)


# Equal axes still return a fresh aliasing view, never the input object.
_IDENTITY_ROWS = [
    ((20, 320, 15), 1, 1),
    ((16, 128, 64, 60), 2, -2),
    ((256,), 0, 0),
    ((), 0, 0),
]
_QUICK_IDENTITY_ROWS = [((20, 320, 15), 1, 1), ((256,), 0, 0), ((), 0, 0)]


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1",
    tu.selected_cases(_IDENTITY_ROWS, quick=_QUICK_IDENTITY_ROWS),
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_swapaxes_identity_axes(shape, axis0, axis1, dtype):
    if flag_gems.vendor_name == "kunlunxin" and not shape:
        pytest.skip(
            "Kunlunxin native scalar swapaxes copies storage instead of aliasing it"
        )
    required_device = _expected_device()
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes(inp, axis0, axis1)

    assert res_out is not inp
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.data_ptr() == inp.data_ptr()
    _assert_result_equal(res_out, ref_out, required_device=required_device)


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1", [((8, 16, 32), 0, 2), ((16, 32, 8, 4), 1, 3)]
)
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_swapaxes_keeps_lazy_conj(shape, axis0, axis1, dtype):
    required_device = _expected_device()
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base.conj()
    ref_inp = ref_base.conj()

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    res_out = flag_gems.swapaxes(inp, axis0, axis1)

    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_conj()
    _assert_result_equal(res_out, ref_out, required_device=required_device)


# The special-value matrix is derived from the supported dtypes, so e4m3fn gets
# the nan-only case while e5m2 also gets inf and mixed nan/inf.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases([d for d in SUPPORTED_DTYPES if d.is_floating_point]),
    quick=[],
)


@pytest.mark.swapaxes
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_swapaxes_special_values(dtype, scenario):
    required_device = _expected_device()
    inp = _make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapaxes(ref_inp, 0, 1)
    res_out = flag_gems.swapaxes(inp, 0, 1)

    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    _assert_result_equal(res_out, ref_out, required_device=required_device)


_BACKWARD_DTYPES = [
    dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point or dtype.is_complex
]


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1",
    tu.selected_cases([((16, 32, 8), 0, 2), ((4, 8, 16, 32), 1, 3)], quick=[]),
)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_swapaxes_backward(shape, axis0, axis1, dtype):
    required_device = _expected_device()
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.swapaxes(ref_inp, axis0, axis1)
    upstream = _make_input(dtype, ref_out.shape, ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    res_out = flag_gems.swapaxes(inp, axis0, axis1)
    assert res_out.requires_grad
    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]

    # Backward permutes the upstream gradient, so it is exact.
    assert res_grad.shape == ref_grad.shape
    _assert_result_equal(res_grad, ref_grad, required_device=required_device)


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1",
    [((2, 3, 4), 0, 3), ((2, 3, 4), -4, 0), ((256,), 0, 2), ((), 1, 0)],
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="Kunlunxin native swapaxes fails the IndexError contract for these invalid axes (including heap corruption)",
)
def test_swapaxes_out_of_range_axes(shape, axis0, axis1):
    inp = _make_input(torch.float32, shape, ["-1", "1"])

    with pytest.raises(IndexError):
        flag_gems.swapaxes(inp, axis0, axis1)


@pytest.mark.swapaxes
@pytest.mark.parametrize("axis0,axis1", [(0.5, 1), (0, None), ("1", 0)])
def test_swapaxes_invalid_axis_type(axis0, axis1):
    inp = _make_input(torch.float32, (2, 3, 4), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.swapaxes(inp, axis0, axis1)


@pytest.mark.swapaxes
def test_swapaxes_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.swapaxes(3.14, 0, 1)


@pytest.mark.swapaxes
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_swapaxes_expanded_input(dtype):
    required_device = _expected_device()
    base = _make_input(dtype, (1, 3), ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp, ref_inp = base.expand(4, 3), ref_base.expand(4, 3)

    ref_out = torch.ops.aten.swapaxes(ref_inp, 0, 1)
    res_out = flag_gems.swapaxes(inp, 0, 1)

    _assert_view_matches(res_out, ref_out, inp, ref_inp)
    _assert_result_equal(res_out, ref_out, required_device=required_device)
