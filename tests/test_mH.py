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
    if result.dtype == torch.complex32:
        result = result.to(torch.complex64)
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    return result.resolve_conj().resolve_neg()


def _comparison_values(result, reference, *, required_device):
    # Check the original ABI before observing values on CPU. Layout and alias
    # assertions at each call site still inspect the original device tensors.
    assert result.device == required_device
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else result.device)
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape and result.layout == reference.layout
    if (
        result.dtype in _FP8_DTYPES
        or result.dtype == torch.complex32
        or (flag_gems.vendor_name in ("ascend", "kunlunxin") and result.is_complex())
        or (flag_gems.vendor_name == "ascend" and result.dtype == torch.int64)
        # Kunlunxin typed transfer/comparison loses the lazy negative flag.
        or (
            flag_gems.vendor_name == "kunlunxin"
            and (result.is_neg() or reference.is_neg())
        )
    ):
        result, reference = _numeric_values(result), _numeric_values(reference)
    return result, reference


def _assert_result_equal(result, reference, *, required_device):
    result, reference = _comparison_values(
        result, reference, required_device=required_device
    )
    tu.assert_result_equal(result, reference)


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name == "kunlunxin" and dtype in _FP8_DTYPES:
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
    if flag_gems.vendor_name == "kunlunxin" and dtype in _FP8_DTYPES:
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


def _fill_value(inp, value):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and inp.device.type != "cpu"
        and inp.is_complex()
    ):
        # Write through the returned view without rebinding storage or flags.
        inp.copy_(
            torch.full(inp.shape, value, dtype=inp.dtype, device="cpu").to(inp.device)
        )
    else:
        inp.fill_(value)


# aten::mH returns a zero-copy conjugate-transpose view: it swaps the last two
# dimensions and toggles the lazy conjugate bit (never materialized; a no-op for
# real dtypes). There is no kernel arithmetic to compare, so values are compared
# exactly and the view contract (shared storage, strides, storage offset,
# conjugate bit, write-through) is asserted separately. Only rank >= 2 is a
# transpose: rank 1 is rejected by aten and 0-D is the deprecated conj() path,
# so the value grid keeps the ranks the operator accepts.
_MH_DTYPES = (
    tu.REQUIRED_DTYPES
    + [torch.bool, torch.complex64]
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])
)

_MH_COMPLEX_DTYPES = [
    dtype for dtype in (torch.complex64, torch.complex128) if dtype in _MH_DTYPES
]

_MH_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2] + [
    (1, 1),
    (1, 7),
    (7, 1),
    (0, 3),
    (3, 0),
    (2, 0, 4),
]

# Inputs that are already transposed, strided or offset windows stay views, so
# the result has to re-derive strides and keep the shared storage offset. Every
# row is a small tensor, so all of them stay in quick mode.
_MH_LAYOUT_ROWS = [
    ((6, 8), "transposed"),
    ((6, 8), "column_step"),
    ((6, 8), "offset_window"),
    ((4, 6, 10), "offset_window"),
]

_MH_WRITE_ROWS = [
    ((4, 6), "asis"),
    ((6, 8), "transposed"),
    ((6, 8), "offset_window"),
]

# "input already conjugated or not" x "input already transposed or not": mH
# toggles the conjugate bit instead of materializing the conjugation.
_MH_CONJ_ROWS = [
    (layout, dtype)
    for layout in ("asis", "transposed", "conj", "transposed_conj")
    for dtype in _MH_COMPLEX_DTYPES
]

_MH_BACKWARD_ROWS = tu.selected_cases(
    [
        (shape, dtype)
        for shape in ((4, 6), (7, 13, 29), (2, 3, 4))
        for dtype in _MH_DTYPES
        if dtype.is_floating_point or dtype.is_complex
    ],
    quick=[],
)

_MH_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_MH_DTYPES), quick=[])

# Rank-1 forms (including the empty (0,) tensor) raise natively.
_MH_REJECT_SHAPES = [(5,), (1,), (0,)]


def _apply_layout(base, layout):
    if layout == "asis":
        return base
    if layout == "transposed":
        return base.transpose(-1, -2)
    if layout == "column_step":
        return base[..., ::2]
    if layout == "offset_window":
        return base[..., 1:4, 1:4]
    if layout == "conj":
        return base.conj()
    if layout == "transposed_conj":
        return base.transpose(-1, -2).conj()
    raise ValueError("unsupported layout " + repr(layout))


def _layout_pair(shape, layout, dtype):
    # Candidate operand, independently stored reference operand, and both
    # backing tensors so a write through either view can be traced back.
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    return _apply_layout(base, layout), _apply_layout(ref_base, layout), base, ref_base


def _assert_view_semantics(res_out, ref_out, inp, ref_inp):
    assert (res_out is inp) == (ref_out is ref_inp)
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()
    # mH aliases its input instead of copying, so shared storage, strides,
    # storage offset and the lazy conjugate bit must match aten exactly.
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _MH_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.mH
@pytest.mark.parametrize(
    "shape,dtype,value_range",
    [
        pytest.param(
            shape,
            dtype,
            value_range,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and str(dtype).startswith("torch.float8_")
                and 0 not in shape,
                reason="Ascend cannot construct nonempty FP8 inputs with device casts or fills",
            ),
            # Keep the original IDs while marking dtype and source shape together.
            id=f"dtype{case_index}-value_range{case_index}-shape{shape_index}",
        )
        for case_index, (dtype, value_range) in enumerate(_DTYPE_RANGE_CASES)
        for shape_index, shape in enumerate(_MH_SHAPES)
    ],
)
def test_mH(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mH(ref_inp)
    required_device = inp.device
    res_out = flag_gems.mH(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.mH
@pytest.mark.parametrize("shape,layout", _MH_LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_mH_strided_input(shape, layout, dtype):
    inp, ref_inp, _, _ = _layout_pair(shape, layout, dtype)

    ref_out = torch.ops.aten.mH(ref_inp)
    required_device = inp.device
    res_out = flag_gems.mH(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.mH
@pytest.mark.parametrize("layout,dtype", _MH_CONJ_ROWS)
def test_mH_conjugate_state(layout, dtype):
    inp, ref_inp, _, _ = _layout_pair((6, 8), layout, dtype)

    ref_out = torch.ops.aten.mH(ref_inp)
    required_device = inp.device
    res_out = flag_gems.mH(inp)

    assert res_out.is_conj() is ref_out.is_conj()
    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.mH
@pytest.mark.parametrize("shape,layout", _MH_WRITE_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_mH_write_through(shape, layout, dtype):
    inp, ref_inp, base, ref_base = _layout_pair(shape, layout, dtype)

    required_device = inp.device
    res_out = flag_gems.mH(inp)
    ref_out = torch.ops.aten.mH(ref_inp)

    # mH shares storage, so a write through the returned view must reach the
    # input and its backing tensor exactly like the native view does.
    _fill_value(res_out, 2.5)
    _fill_value(ref_out, 2.5)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_result_equal(inp, ref_inp, required_device=required_device)
    _assert_result_equal(base, ref_base, required_device=required_device)


@pytest.mark.mH
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and str(dtype).startswith("torch.float8_"),
                reason="Ascend cannot construct FP8 inputs with device casts or fills",
            ),
        )
        for dtype in _MH_DTYPES
    ],
)
def test_mH_0d_scalar(dtype):
    # aten::mH warns and degrades to conj() on 0-D input: a real tensor comes
    # back as itself, a complex one as a lazy conjugate view on the same storage.
    inp = _make_input(dtype, (), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mH(ref_inp)
    required_device = inp.device
    res_out = flag_gems.mH(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    assert (res_out is inp) == (ref_out is ref_inp)
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.mH
@pytest.mark.parametrize(
    "shape,dtype",
    [
        pytest.param(
            shape,
            dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and str(dtype).startswith("torch.float8_"),
                reason="Ascend cannot construct FP8 inputs with device casts or fills",
            ),
        )
        for shape, dtype in _MH_BACKWARD_ROWS
    ],
)
def test_mH_backward(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)
    upstream = _make_input(dtype, shape[:-2] + (shape[-1], shape[-2]), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.mH(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]

    required_device = inp.device
    res_out = flag_gems.mH(inp)
    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)

    # The adjoint of a transpose/conjugate view is the same relayout applied to
    # the upstream gradient, so it is compared exactly.
    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    _assert_result_equal(res_in_grad, ref_in_grad, required_device=required_device)


@pytest.mark.mH
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and str(dtype).startswith("torch.float8_"),
                reason="Ascend cannot construct FP8 inputs with device casts or fills",
            ),
        )
        for dtype, scenario in _MH_SPECIAL_CASES
    ],
)
def test_mH_special_values(dtype, scenario):
    inp = _make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mH(ref_inp)
    required_device = inp.device
    res_out = flag_gems.mH(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.mH
@pytest.mark.parametrize("shape", _MH_REJECT_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_mH_rejects_1d(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.mH(inp)
