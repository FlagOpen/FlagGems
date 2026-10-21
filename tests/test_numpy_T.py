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
    # Decode before resolving lazy bits: CPU FP8/C32 arithmetic is limited.
    if result.dtype in _FP8_DTYPES:
        result = result.float()
    if result.dtype == torch.complex32:
        result = result.to(torch.complex64)
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    return result.resolve_conj().resolve_neg()


def _assert_result_equal(result, reference, *, required_device):
    # Validate the original ABI before observing numerical values on CPU.
    assert result.device == required_device
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else result.device)
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape and result.layout == reference.layout
    if (
        result.dtype in _FP8_DTYPES
        or result.dtype == torch.complex32
        or (flag_gems.vendor_name in ("ascend", "kunlunxin") and result.is_complex())
        or (flag_gems.vendor_name == "ascend" and result.dtype == torch.int64)
        or (flag_gems.vendor_name == "iluvatar" and result.dtype == torch.float64)
        or (
            flag_gems.vendor_name == "kunlunxin"
            and (result.is_neg() or reference.is_neg())
        )
    ):
        result, reference = _numeric_values(result), _numeric_values(reference)
    tu.assert_result_equal(result, reference)


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


# aten::numpy_T is a zero-copy metadata view (Tensor(a) -> Tensor(a)): it
# reverses every dimension, so 0-D/1-D keep their layout, 2-D swaps both
# dimensions and n-D is a full reversal. It never conjugates -- a lazily
# conjugated complex input keeps its conj bit (only .H/.adjoint toggles it) and
# a negative view keeps its neg bit. The result always aliases the input
# storage, so the tests compare values/shape/dtype through the shared helpers
# and add the alias facts those helpers cannot observe: storage offset,
# view-ness, lazy bits, data_ptr sharing and write-through mutation. The
# operator is unary and takes no scalar operand, so the spec's broadcast and
# scalar/tensor dimensions do not apply.

_NUMPY_T_DTYPES = (
    list(tu.REQUIRED_DTYPES)
    + [torch.bool, torch.complex64, torch.complex32, torch.int16]
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])
)


def _assert_view_result(res_out, ref_out, inp, ref_inp, *, required_device):
    """Shared value comparison plus the alias facts it cannot observe."""
    _assert_result_equal(res_out, ref_out, required_device=required_device)
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    assert res_out._is_view() == ref_out._is_view()
    assert res_out is not inp
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res_out.untyped_storage()._cdata == inp.untyped_storage()._cdata
    assert inp.shape == ref_inp.shape
    assert inp.stride() == ref_inp.stride()
    assert inp.storage_offset() == ref_inp.storage_offset()


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    pytest.param(
        dtype,
        value_range,
    )
    for dtype in _NUMPY_T_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.numpy_T
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,value_range",
    _DTYPE_RANGE_CASES,
)
def test_numpy_T_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.numpy_T(ref_inp)
    required_device = inp.device
    res_out = flag_gems.numpy_T(inp)

    _assert_view_result(res_out, ref_out, inp, ref_inp, required_device=required_device)


_RANK_CASES = [
    ((), ()),
    ((1,), (1,)),
    ((256,), (256,)),
    ((3, 5), (5, 3)),
    ((2, 3, 4), (4, 3, 2)),
    ((16, 7, 57), (57, 7, 16)),
    ((2, 3, 4, 5), (5, 4, 3, 2)),
    ((16, 7, 57, 32, 29), (29, 32, 57, 7, 16)),
    ((0, 3), (3, 0)),
    ((3, 0), (0, 3)),
    ((2, 0, 4), (4, 0, 2)),
    ((1, 1, 1), (1, 1, 1)),
]


@pytest.mark.numpy_T
@pytest.mark.parametrize("shape,expected", _RANK_CASES)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.int64, torch.bool, torch.complex64],
)
def test_numpy_T_rank_boundaries(shape, expected, dtype):
    # Every rank is accepted, including 0-D and zero-element shapes that the
    # seven-shape grid does not reach; the expected shape is the full reversal.
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.numpy_T(ref_inp)
    required_device = inp.device
    res_out = flag_gems.numpy_T(inp)

    assert res_out.shape == expected
    _assert_view_result(res_out, ref_out, inp, ref_inp, required_device=required_device)
    # Even the 0-D "identity" is a view: the native op never hands back the
    # input object itself.
    assert res_out is not inp


def _strided_view(tensor, kind):
    """Build the non-contiguous base forms a view op has to forward."""
    if kind == "transposed":
        return tensor.t()
    if kind == "strided_slice":
        return tensor[:, ::2]
    if kind == "offset_window":
        return tensor[2:6, 1:9]
    if kind == "permuted":
        return tensor.permute(2, 0, 1)
    if kind == "expanded":
        return tensor.expand(4, 3)
    raise AssertionError(f"unknown kind: {kind}")


_STRIDED_CASES = [
    ((6, 10), "transposed"),
    ((10, 10), "strided_slice"),
    ((12, 12), "offset_window"),
    ((8, 16, 4), "permuted"),
    ((1, 3), "expanded"),
]


@pytest.mark.numpy_T
@pytest.mark.parametrize("base_shape,kind", _STRIDED_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.complex64])
def test_numpy_T_noncontiguous_base(base_shape, kind, dtype):
    # Non-unit strides, a nonzero storage offset and a stride-0 broadcast
    # dimension: the reversal must keep the input's own layout instead of
    # compacting it.
    base = _make_input(dtype, base_shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = _strided_view(base, kind)
    ref_inp = _strided_view(ref_base, kind)

    ref_out = torch.ops.aten.numpy_T(ref_inp)
    required_device = inp.device
    res_out = flag_gems.numpy_T(inp)

    _assert_view_result(res_out, ref_out, inp, ref_inp, required_device=required_device)


def _stateful_view(tensor, kind):
    """Attach the lazy state bit (conj / neg) the reversal must preserve."""
    if kind == "conj":
        return tensor.conj()
    if kind == "neg":
        return torch._neg_view(tensor)
    if kind == "neg_slice":
        return torch._neg_view(tensor[:, ::2])
    raise AssertionError(f"unknown kind: {kind}")


_STATE_CASES = [
    ("conj_2d", torch.complex64, "conj", (4, 6)),
    ("conj_3d", torch.complex64, "conj", (3, 5, 7)),
    ("neg_2d", torch.float32, "neg", (4, 6)),
    ("neg_sliced", torch.float32, "neg_slice", (4, 12)),
]


@pytest.mark.numpy_T
@pytest.mark.parametrize(
    "name,dtype,kind,shape",
    _STATE_CASES,
)
def test_numpy_T_preserves_lazy_state(name, dtype, kind, shape):
    del name  # the row id documents which lazy bit the case exercises
    base = _make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = _stateful_view(base, kind)
    ref_inp = _stateful_view(ref_base, kind)

    ref_out = torch.ops.aten.numpy_T(ref_inp)
    required_device = inp.device
    res_out = flag_gems.numpy_T(inp)

    # The reversal must not materialize or drop the lazy bit, and the values of
    # a lazy conjugated view differ from those of the raw storage.
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    _assert_view_result(res_out, ref_out, inp, ref_inp, required_device=required_device)


# Writing through the result must update the original input storage.
_MUTATION_CASES = [
    ((5, 7), torch.float32),
    ((12, 4), torch.int32),
    ((2, 3, 4), torch.float64),
    ((5, 7), torch.uint8),
]


@pytest.mark.numpy_T
@pytest.mark.parametrize(
    "shape,dtype",
    [
        pytest.param(
            shape,
            dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name in ("iluvatar", "kunlunxin")
                and shape == (2, 3, 4)
                and dtype == torch.float64,
                reason="Native float64 mutation cannot preserve the requested payload on this backend",
            ),
        )
        for shape, dtype in _MUTATION_CASES
    ],
)
def test_numpy_T_writes_through_to_input(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.numpy_T(ref_inp)
    required_device = inp.device
    res_out = flag_gems.numpy_T(inp)
    _assert_view_result(res_out, ref_out, inp, ref_inp, required_device=required_device)

    res_out.fill_(3)
    ref_out.fill_(3)

    # One storage behind both: a write through the reversed view has to appear
    # in the original tensor, exactly as it does for the native op.
    _assert_result_equal(inp, ref_inp, required_device=required_device)


# Backward is default-only; the compact rank/layout/state/mutation/overload rows
# above keep the quick suite meaningful without autograd groups.
_BACKWARD_CASES = tu.selected_cases(
    [(16, 64), (7, 13, 29), (4, 8, 16, 60), (16, 7, 57, 32, 29)],
    quick=[],
)


@pytest.mark.numpy_T
@pytest.mark.parametrize("shape", _BACKWARD_CASES)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
        )
        for dtype in _NUMPY_T_DTYPES
        if dtype.is_floating_point or dtype.is_complex
    ],
)
def test_numpy_T_backward(shape, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"]).detach().requires_grad_(True)
    ref_inp = tu.to_reference(inp).detach().requires_grad_(True)
    upstream = _make_input(dtype, tuple(reversed(shape)), ["-1", "1"])
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.numpy_T(ref_inp)
    required_device = inp.device
    res_out = flag_gems.numpy_T(inp)

    assert res_out.requires_grad == ref_out.requires_grad
    _assert_view_result(res_out, ref_out, inp, ref_inp, required_device=required_device)

    # Differentiate through the original leaf: the gradient of a pure dimension
    # reversal is that same reversal applied to the upstream gradient, i.e. an
    # exact relayout with no arithmetic.
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]
    _assert_result_equal(res_grad, ref_grad, required_device=required_device)


# Positive special-value cases are default-only; the shared generator decides
# which scenarios each dtype can represent (e4m3fn only holds nan, e5m2 holds
# nan/inf/mixed). Two ranks cover the swap and the full-reversal paths, and the
# one-element axes keep the -1/0/1 and -inf/0/+inf rows distinct.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_NUMPY_T_DTYPES), quick=[])
_SPECIAL_SHAPES = [(1, 5), (1, 1, 5)]


@pytest.mark.numpy_T
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
        )
        for dtype, scenario in _SPECIAL_CASES
    ],
)
@pytest.mark.parametrize("shape", _SPECIAL_SHAPES)
def test_numpy_T_special_values(dtype, scenario, shape):
    inp = _make_special_input(dtype, scenario).reshape(shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.numpy_T(ref_inp)
    required_device = inp.device
    res_out = flag_gems.numpy_T(inp)

    _assert_view_result(res_out, ref_out, inp, ref_inp, required_device=required_device)


# Both runtime overloads of aten::numpy_T have the same semantics; the candidate
# is reached through the one public entry point for each of them.
_OVERLOAD_SHAPES = [(), (5,), (2, 3), (2, 3, 4)]


@pytest.mark.numpy_T
@pytest.mark.parametrize("shape", _OVERLOAD_SHAPES)
@pytest.mark.parametrize("overload", ["default", "a"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_numpy_T_overload_call_forms(shape, overload, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = getattr(torch.ops.aten.numpy_T, overload)(ref_inp)
    required_device = inp.device
    res_out = flag_gems.numpy_T(inp)

    _assert_view_result(res_out, ref_out, inp, ref_inp, required_device=required_device)


# numpy_T accepts every dtype and every rank up to the framework's MAX_DIMS and
# takes no parameters, so there is no unsupported dtype/dim/parameter value to
# reject; only invalid argument shapes remain. Both rows stay in default and
# quick mode because the negative dimension is required in either level.
@pytest.mark.numpy_T
@pytest.mark.parametrize(
    "bad_input",
    [
        pytest.param([[1.0, 2.0]], id="python_nested_list"),
        pytest.param(3.0, id="python_float"),
        pytest.param(None, id="none"),
    ],
)
def test_numpy_T_rejects_non_tensor_argument(bad_input):
    # A python candidate may fail with AttributeError while a schema-checked
    # entry point raises TypeError/RuntimeError; either way the candidate must
    # reject the argument instead of silently building a view.
    with pytest.raises((RuntimeError, TypeError, AttributeError)):
        flag_gems.numpy_T(bad_input)


@pytest.mark.numpy_T
def test_numpy_T_rejects_wrong_argument_count():
    inp = torch.zeros(2, 3, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError, AttributeError)):
        flag_gems.numpy_T(inp, 1)
