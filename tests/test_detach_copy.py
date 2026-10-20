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
pytestmark = pytest.mark.detach_copy


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name not in ("ascend", "kunlunxin") or dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return tu.make_input(dtype, shape, value_range)
    # Encode FP8 on CPU; device-side FP32-to-FP8 casts are unavailable.
    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)
    if low == high:
        result = torch.full(shape, low, dtype=dtype, device="cpu")
    else:
        result = torch.testing.make_tensor(
            shape, dtype=dtype, device="cpu", low=low, high=high
        )
    return result.to(flag_gems.device)


def _make_special_input(dtype, scenario):
    if flag_gems.vendor_name not in ("ascend", "kunlunxin") or dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return tu.make_special_input(dtype, scenario)
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    return (
        torch.tensor(payloads[scenario], device="cpu", dtype=torch.float32)
        .to(dtype)
        .to(flag_gems.device)
    )


def _make_out_buffer(shape, dtype, device):
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and (
        torch.device(device).type == "cpu"
        or flag_gems.vendor_name in ("ascend", "kunlunxin")
    ):
        # Older CPU builds have FP8 casts but no full/fill kernel.
        return (
            torch.full(shape, 7, dtype=torch.float32, device="cpu").to(dtype).to(device)
        )
    creation_device = device
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype.is_complex:
        # Device full kernels do not cover complex payloads.
        creation_device = "cpu"
    return torch.full(shape, 7, dtype=dtype, device=creation_device).to(device)


def _comparison_to_cpu(tensor):
    fp8 = tensor.dtype in tu._FP8_DTYPES
    if tensor.layout == torch.strided and (
        flag_gems.vendor_name in ("ascend", "kunlunxin")
        and tensor.device.type != "cpu"
        or (fp8 and tensor.is_neg())
    ):
        values = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
            tensor.untyped_storage().cpu(),
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
        )
        if fp8:
            values = values.float()
        if tensor.is_conj():
            values = values.conj()
        if tensor.is_neg():
            values = torch._neg_view(values)
        return values
    values = tensor.cpu()
    return values.float() if fp8 else values


def _assert_result_equal(result, reference):
    # Validate the original device and metadata before CPU value observation.
    device = torch.device(flag_gems.device)
    if device.index is None and device.type != "cpu":
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    assert result.device == device
    assert result.dtype == reference.dtype
    assert result.layout == reference.layout
    assert result.shape == reference.shape
    if utils.TO_CPU:
        assert reference.device == torch.device("cpu")
    else:
        assert result.device == reference.device
    if result.dtype in tu._FP8_DTYPES or (
        flag_gems.vendor_name in ("ascend", "kunlunxin") and result.is_complex()
    ):
        result, reference = _comparison_to_cpu(result), _comparison_to_cpu(reference)
    tu.assert_result_equal(result, reference)


# Copy values into independent contiguous storage; backward is unsupported.
_DETACH_COPY_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.bool,
        torch.complex64,
    ]
)

_DETACH_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_DETACH_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
_DETACH_COPY_NO_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]
_DETACH_COPY_STORAGE_SHAPES = [(16, 32), (64, 128)]


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp):
    assert res_out.device == inp.device
    assert res_out.is_contiguous()
    assert res_out.stride() == ref_out.stride()
    # Zero-element tensors carry a null data pointer on every tensor, so the
    # no-alias check is only meaningful for non-empty inputs.
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    _assert_result_equal(res_out, ref_out)
    # The input must be untouched by the copy.
    _assert_result_equal(inp, ref_inp)


_DETACH_COPY_INPUT_DTYPE_CASES = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend"
            and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
            reason="Ascend cannot construct or compare FP8 payloads with device kernels",
        ),
    )
    for dtype in _DETACH_COPY_DTYPES
]


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DETACH_COPY_INPUT_DTYPE_CASES)
def test_detach_copy(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    # to_reference creates an independent snapshot for the input mutation check.
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    pytest.param(
        dtype,
        value_range,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend"
            and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
            reason="Ascend cannot construct or compare FP8 payloads with device kernels",
        ),
    )
    for dtype in _DETACH_COPY_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8 or value_range in (["-1", "1"], ["0", "max"])
]


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype, value_range",
    _DTYPE_RANGE_CASES,
)
def test_detach_copy_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_detach_copy_special_values(dtype):
    inp = torch.tensor(
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            0.0,
            -0.0,
            1.5,
            -2.5,
            1e30,
            -1e30,
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_result_equal(res_out, ref_out)
    # -0.0 must copy with its sign bit intact (equal_nan-tolerant compares treat
    # -0.0 == 0.0, so pin the sign explicitly).
    # Read bits on CPU: Ascend signbit does not distinguish -0.0 from +0.0.
    assert res_out.device == inp.device
    assert torch.equal(torch.signbit(res_out.cpu()), torch.signbit(ref_out.cpu()))


@pytest.mark.detach_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DETACH_COPY_INPUT_DTYPE_CASES)
def test_detach_copy_out(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out = _make_out_buffer(shape, ref_inp.dtype, ref_inp.device)
    res_out = _make_out_buffer(shape, dtype, flag_gems.device)

    ref_ret = torch.ops.aten.detach_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.detach_copy(inp, out=res_out)

    # The .out overload must write into and return the caller's buffer.
    assert res_ret is res_out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp)


@pytest.mark.detach_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype, value_range",
    _DTYPE_RANGE_CASES,
)
def test_detach_copy_out_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = _make_out_buffer(shape, ref_inp.dtype, ref_inp.device)
    res_out = _make_out_buffer(shape, dtype, flag_gems.device)

    ref_ret = torch.ops.aten.detach_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.detach_copy(inp, out=res_out)

    assert res_ret is res_out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_INPUT_DTYPE_CASES)
def test_detach_copy_non_contiguous(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        pytest.skip("Kunlunxin detach_copy cannot copy noncontiguous FP8 tensors")
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base.transpose(-1, -2)
    ref_inp = ref_base.transpose(-1, -2)
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_empty(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_STORAGE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_detach_copy_independent_storage(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_result_equal(res_out, ref_out)
    res_out.fill_(3.25)
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    _assert_result_equal(inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_NO_BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES, quick=utils.FLOAT_DTYPES)
)
def test_detach_copy_no_backward(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    res_out = flag_gems.detach_copy(inp)
    _assert_result_equal(res_out, ref_out)
    _assert_result_equal(inp, ref_inp)

    with pytest.raises(RuntimeError):
        torch.autograd.grad(res_out, inp, grad_outputs=grad)


@pytest.mark.detach_copy
def test_detach_copy_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.detach_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.detach_copy(3.14)


@pytest.mark.detach_copy_out
def test_detach_copy_out_rejects_wrong_dtype():
    inp = _make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_out_bad = torch.empty(8, dtype=torch.int32, device=flag_gems.device)
    res_out_bad = torch.empty(8, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten.detach_copy.out(ref_inp, out=ref_out_bad)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.detach_copy(inp, out=res_out_bad)


@pytest.mark.detach_copy
@pytest.mark.parametrize(
    "dtype, scenario",
    [
        pytest.param(
            dtype,
            scenario,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
                reason="Ascend cannot construct or compare FP8 payloads with device kernels",
            ),
        )
        for dtype, scenario in tu.selected_cases(
            tu.special_value_cases(_DETACH_COPY_DTYPES)
        )
    ],
)
def test_detach_copy_special_scenarios(dtype, scenario):
    inp = _make_special_input(dtype, scenario)
    reference = tu.to_reference(inp)
    expected = torch.ops.aten.detach_copy(reference)
    actual = flag_gems.detach_copy(inp)
    _assert_result_equal(actual, expected)
