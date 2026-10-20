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

# aten::is_inference(Tensor self) -> bool reads the inference flag of one
# TensorImpl and returns a Python bool. The flag is a per-tensor property, so the
# tested contract is only that flag_gems.is_inference reports the same state as
# torch.ops.aten.is_inference for the same kind of operand.


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
    cpu_encoding = cpu_encoding or (
        flag_gems.vendor_name == "ascend" and dtype == torch.float64
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


_KUNLUN_FP64_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="Kunlunxin narrows float64 storage to float32",
)


def _dtype_marks(dtype):
    return (_KUNLUN_FP64_SKIP,) if dtype == torch.float64 else ()


def _check_pair(inp, reference):
    assert inp.device == _device(flag_gems.device)
    expected = "cpu" if utils.TO_CPU else flag_gems.device
    assert reference.device == _device(expected)
    assert inp.dtype == reference.dtype and inp.shape == reference.shape
    assert inp.stride() == reference.stride()
    assert inp.storage_offset() == reference.storage_offset()
    assert (inp.is_conj(), inp.is_neg()) == (reference.is_conj(), reference.is_neg())
    assert torch.ops.aten.is_inference(inp) is torch.ops.aten.is_inference(reference)


def _reference_input(inp):
    if utils.TO_CPU:
        with torch.inference_mode(torch.ops.aten.is_inference(inp)):
            return _copy_to(inp, "cpu")
    return inp


_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
    torch.bool,
    torch.complex64,
]

_FLOAT_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

_BOUNDARY_DTYPES = [torch.float32, torch.int64, torch.bool]

# Scalar and empty tensors retain their allocation-context state.
_BOUNDARY_SHAPES = [(), (0,), (2, 0)]

# Positive special-value workloads are default-only: quick selects none of them.
_SPECIAL_ROWS = tu.selected_cases(
    [
        (inference, dtype, scenario)
        for inference in (False, True)
        for dtype, scenario in tu.special_value_cases(_FLOAT_DTYPES)
    ],
    quick=[],
)

_VIEW_FORMS = ("view", "transpose", "slice", "expand", "detach")


def _state_pair(base, inference):
    """Two independent operands carrying the requested inference state.

    A clone created inside torch.inference_mode() is an inference tensor, while
    a clone, a device move or a storage copy outside the context (for example
    the one tu.to_reference builds) clears the flag. Both operands are therefore
    allocated under the same context, including the independent CPU reference.
    """

    def clone_pair():
        inp = base.clone()
        reference = _copy_to(base, "cpu") if utils.TO_CPU else base.clone()
        assert inp.dtype == base.dtype and inp.shape == base.shape
        _check_pair(inp, reference)
        return inp, reference

    if inference:
        with torch.inference_mode():
            return clone_pair()
    return clone_pair()


def _make_view(tensor, form):
    if form == "view":
        return tensor.view(2, 12)
    if form == "transpose":
        return tensor.t()
    if form == "slice":
        return tensor[1:3]
    if form == "expand":
        return tensor[0:1].expand(3, 6)
    return tensor.detach()


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


@pytest.mark.is_inference
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,value_range",
    [
        pytest.param(dtype, value_range, marks=_dtype_marks(dtype))
        for dtype, value_range in _DTYPE_RANGE_CASES
    ],
)
@pytest.mark.parametrize("inference", [False, True])
def test_is_inference_state(dtype, value_range, shape, inference):
    base = _make_input(dtype, shape, value_range)
    inp, ref_inp = _state_pair(base, inference)

    _check_pair(inp, ref_inp)
    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)

    assert isinstance(res_out, bool)
    assert res_out is ref_out


@pytest.mark.is_inference
@pytest.mark.parametrize("shape", _BOUNDARY_SHAPES)
@pytest.mark.parametrize("dtype", _BOUNDARY_DTYPES)
@pytest.mark.parametrize("inference", [False, True])
def test_is_inference_scalar_and_empty(dtype, shape, inference):
    base = _make_input(dtype, shape, ["-1", "1"])
    inp, ref_inp = _state_pair(base, inference)

    _check_pair(inp, ref_inp)
    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)

    assert isinstance(res_out, bool)
    assert res_out is ref_out


@pytest.mark.is_inference
@pytest.mark.parametrize("dtype", _BOUNDARY_DTYPES)
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("query_context", [False, True])
def test_is_inference_context_is_not_tensor_state(dtype, inference, query_context):
    base = _make_input(dtype, (4, 6), ["-1", "1"])
    inp, ref_inp = _state_pair(base, inference)
    # Allocation determines the tensor flag; the query context must not change it.
    _check_pair(inp, ref_inp)
    with torch.inference_mode(query_context):
        ref = torch.ops.aten.is_inference(ref_inp)
        res = flag_gems.is_inference(inp)
    assert isinstance(res, bool)
    assert res is ref


@pytest.mark.is_inference
@pytest.mark.parametrize("form", _VIEW_FORMS)
@pytest.mark.parametrize("dtype", _BOUNDARY_DTYPES)
@pytest.mark.parametrize("clone_after_view", [False, True])
def test_is_inference_derived_state(dtype, form, clone_after_view):
    base = _make_input(dtype, (4, 6), ["-1", "1"])
    inferred, ref_inferred = _state_pair(base, True)

    # Views and detach share the source storage and inherit its flag ...
    inp = _make_view(inferred, form)
    ref_inp = _make_view(ref_inferred, form)
    if clone_after_view:
        # A clone made outside inference mode starts an ordinary tensor.
        inp = inp.clone()
        # Native clones can choose different strides on CPU and the device.
        # Snapshot the actual device view before the candidate reads it.
        ref_inp = _copy_to(inp, "cpu") if utils.TO_CPU else ref_inp.clone()
    _check_pair(inp, ref_inp)
    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)
    assert isinstance(res_out, bool)
    assert res_out is ref_out


@pytest.mark.is_inference
@pytest.mark.parametrize("dtype", _BOUNDARY_DTYPES)
def test_is_inference_no_grad_is_not_inference(dtype):
    with torch.no_grad():
        inp = _make_input(dtype, (4, 6), ["-1", "1"])
        ref_inp = _reference_input(_make_input(dtype, (4, 6), ["-1", "1"]))

    _check_pair(inp, ref_inp)
    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)

    assert isinstance(res_out, bool)
    assert res_out is ref_out
    assert res_out is False


@pytest.mark.is_inference
@pytest.mark.parametrize("inference", [False, True])
def test_is_inference_does_not_mutate_input(inference):
    base = _make_input(torch.float32, (4, 6), ["-1", "1"])
    inp, ref_inp = _state_pair(base, inference)
    inp = inp.t()
    ref_inp = ref_inp.t()
    storage_before = inp.untyped_storage().data_ptr()
    layout_before = (
        inp.dtype,
        inp.device,
        inp.data_ptr(),
        inp.storage_offset(),
        tuple(inp.shape),
        tuple(inp.stride()),
    )

    _check_pair(inp, ref_inp)
    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)

    assert isinstance(res_out, bool)
    assert res_out is ref_out
    # The query is read-only: no storage swap, no metadata change, no materiali-
    # zation and no change to the reported state.
    assert inp.untyped_storage().data_ptr() == storage_before
    assert (
        inp.dtype,
        inp.device,
        inp.data_ptr(),
        inp.storage_offset(),
        tuple(inp.shape),
        tuple(inp.stride()),
    ) == layout_before
    assert torch.ops.aten.is_inference(inp) is ref_out
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.is_inference
@pytest.mark.parametrize(
    "inference,dtype,scenario",
    [
        pytest.param(inference, dtype, scenario, marks=_dtype_marks(dtype))
        for inference, dtype, scenario in _SPECIAL_ROWS
    ],
)
def test_is_inference_special_values(inference, dtype, scenario):
    # NaN/Inf payloads must not change the reported state. e4m3fn only appears
    # with the nan scenario because it cannot represent infinity.
    inp, ref_inp = _state_pair(_make_special_input(dtype, scenario), inference)

    _check_pair(inp, ref_inp)
    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)

    assert isinstance(res_out, bool)
    assert res_out is ref_out


@pytest.mark.is_inference
@pytest.mark.parametrize("bad_input", [3.14, 1, True, [1, 2], "x"])
def test_is_inference_negative_non_tensor(bad_input):
    # None is deliberately absent: aten accepts it as an undefined tensor and
    # returns False instead of rejecting it, so it is not an invalid-input case.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_inference(bad_input)


@pytest.mark.is_inference
@pytest.mark.parametrize("second_arg", [1, 0, 2.0, True, (1,)])
def test_is_inference_negative_extra_positional(second_arg):
    inp = _make_input(torch.float32, (4, 6), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_inference(inp, second_arg)


@pytest.mark.is_inference
def test_is_inference_negative_missing_argument():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_inference()


@pytest.mark.is_inference
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_is_inference_negative_out_kwarg(dtype):
    # aten::is_inference has no out= overload.
    inp = _make_input(dtype, (4, 6), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_inference(inp, out=inp)


@pytest.mark.is_inference
def test_is_inference_undefined_tensor():
    ref = torch.ops.aten.is_inference(None)
    res = flag_gems.is_inference(None)

    assert isinstance(res, bool)
    assert res is ref
