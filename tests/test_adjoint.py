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


def _assert_result_close(result, reference, *, required_device, atol=1e-4):
    dtype = result.dtype
    result, reference = _comparison_values(
        result, reference, required_device=required_device
    )
    if dtype.is_floating_point or dtype.is_complex:
        if flag_gems.vendor_name == "kunlunxin" and dtype in _FP8_DTYPES:
            # Same original-dtype policy as the shared non-Kunlunxin table;
            # the Kunlunxin table omits FP8 despite these declared cases.
            rtol = 1e-3
        else:
            rtol = flag_gems.testing.RESOLUTION[dtype]
        torch.testing.assert_close(
            result, reference, rtol=rtol, atol=atol, equal_nan=True
        )
    else:
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


_ASCEND_FP8_INPUT_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend cannot generate nonempty FP8 test payloads",
)


def _input_marks(dtype):
    if dtype.is_complex:
        return (_ASCEND_COMPLEX_SKIP,)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return (_ASCEND_FP8_INPUT_SKIP,)
    return ()


_ASCEND_COMPLEX_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend complex input/reference support is incomplete",
)


# adjoint swaps the last two dimensions and toggles the conjugate bit for
# complex inputs. The result shares storage with the input.
_ADJOINT_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16, torch.complex32])
    + [torch.complex64, torch.bool]
)
# Scalar and 1-D behavior is covered separately below.
_ADJOINT_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]


def _transposed_shape(shape):
    # Shape of adjoint(x): the last two dimensions are swapped.
    if len(shape) >= 2:
        return shape[:-2] + (shape[-1], shape[-2])
    return shape


def _assert_view_semantics(res_out, ref_out, inp):
    # adjoint returns an aliasing view (Tensor(a)): shape, strides, storage
    # offset, conjugation state and the shared storage must match aten exactly.
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _ADJOINT_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(dtype, marks=_input_marks(dtype)) for dtype in _ADJOINT_DTYPES],
)
def test_adjoint(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    required_device = inp.device
    res_out = flag_gems.adjoint(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _ADJOINT_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _ADJOINT_SHAPES)
@pytest.mark.parametrize(
    "dtype,value_range",
    [
        pytest.param(dtype, value_range, marks=_input_marks(dtype))
        for dtype, value_range in _DTYPE_RANGE_CASES
    ],
)
def test_adjoint_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    required_device = inp.device
    res_out = flag_gems.adjoint(inp)

    _assert_view_semantics(res_out, ref_out, inp)
    _assert_result_equal(res_out, ref_out, required_device=required_device)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", [(8, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(dtype, marks=_input_marks(dtype)) for dtype in _ADJOINT_DTYPES],
)
def test_adjoint_non_contiguous(shape, dtype):
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.adjoint(ref_inp)
    required_device = inp.device
    res_out = flag_gems.adjoint(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", [(16, 32), (4, 8, 16)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_input_marks(dtype))
        for dtype in utils.FLOAT_DTYPES + utils.COMPLEX_DTYPES
    ],
)
def test_adjoint_toggle(shape, dtype):
    # An already-conjugated input must produce an unconjugated view.
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)

    inp = torch.ops.aten.adjoint(base)
    ref_inp = torch.ops.aten.adjoint(ref_base)
    assert inp.is_conj() == ref_inp.is_conj()

    ref_out = torch.ops.aten.adjoint(ref_inp)
    required_device = inp.device
    res_out = flag_gems.adjoint(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, base)
    assert not res_out.is_conj()


@pytest.mark.adjoint
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_input_marks(dtype))
        for dtype in tu.selected_cases(utils.ALL_FLOAT_DTYPES)
    ],
)
def test_adjoint_special_values(dtype):
    values = torch.tensor(
        [
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
            ]
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = tu.to_reference(values)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    required_device = values.device
    res_out = flag_gems.adjoint(values)

    _assert_view_semantics(res_out, ref_out, values)
    _assert_result_equal(res_out, ref_out, required_device=required_device)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", [(16, 32), (4, 8, 16)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adjoint_mutation(shape, dtype):
    # Writing through the returned view must update the input.
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    required_device = inp.device
    res_out = flag_gems.adjoint(inp)
    ref_out = torch.ops.aten.adjoint(ref_inp)

    res_out.fill_(2.5)
    ref_out.fill_(2.5)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    assert res_out.data_ptr() == inp.data_ptr()
    _assert_result_equal(inp, ref_inp, required_device=required_device)


@pytest.mark.adjoint
@pytest.mark.parametrize("shape", [(16, 64), (7, 13, 29)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_input_marks(dtype))
        for dtype in tu.selected_cases(
            [d for d in _ADJOINT_DTYPES if d.is_floating_point or d.is_complex]
        )
    ],
)
def test_adjoint_backward(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = _make_input(dtype, _transposed_shape(shape), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)[0]

    required_device = inp.device
    res_out = flag_gems.adjoint(inp)
    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_view_semantics(res_out, ref_out, inp)

    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad)[0]
    _assert_result_close(res_in_grad, ref_in_grad, required_device=required_device)


@pytest.mark.adjoint
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(dtype, marks=_input_marks(dtype)) for dtype in _ADJOINT_DTYPES],
)
def test_adjoint_0d(dtype):
    # ATen accepts scalars as a lazy conj(), with a deprecation warning.
    inp = _make_input(dtype, (), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    required_device = inp.device
    res_out = flag_gems.adjoint(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    assert res_out.is_conj() == ref_out.is_conj()


@pytest.mark.adjoint
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(dtype, marks=_input_marks(dtype))
        for dtype in [torch.float32, torch.complex64]
    ],
)
def test_adjoint_1d_raises(dtype):
    inp = _make_input(dtype, (5,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.ops.aten.adjoint(ref_inp)
    with pytest.raises(RuntimeError):
        flag_gems.adjoint(inp)


@pytest.mark.adjoint
def test_adjoint_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.adjoint(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.adjoint(3.14)


@pytest.mark.adjoint
@pytest.mark.parametrize(
    "dtype, scenario",
    [
        pytest.param(dtype, scenario, marks=_input_marks(dtype))
        for dtype, scenario in tu.selected_cases(
            tu.special_value_cases(_ADJOINT_DTYPES)
        )
    ],
)
def test_adjoint_special_scenarios(dtype, scenario):
    inp = _make_special_input(dtype, scenario)
    inp = inp.reshape(1, -1)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.adjoint(ref_inp)
    required_device = inp.device
    res_out = flag_gems.adjoint(inp)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
