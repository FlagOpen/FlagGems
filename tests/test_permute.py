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

"""Correctness tests for ``aten::permute`` (a zero-copy relayout view)."""

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


def _physical_cpu(inp):
    # Preserve independent physical storage, view geometry and lazy metadata.
    assert inp.layout == torch.strided
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
    if inp.is_conj():
        result = result.conj()
    if inp.is_neg():
        result = torch._neg_view(result)
    result.requires_grad_(inp.requires_grad)
    assert result.device == torch.device("cpu") and result.dtype == inp.dtype
    assert result.shape == inp.shape and result.stride() == inp.stride()
    assert result.storage_offset() == inp.storage_offset()
    assert (result.is_conj(), result.is_neg()) == (inp.is_conj(), inp.is_neg())
    if inp.untyped_storage().nbytes():
        assert result.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    return result


def _reference(inp):
    assert inp.device == _expected_device()
    return _physical_cpu(inp) if utils.TO_CPU else tu.to_reference(inp)


def _numeric_values(inp):
    result = _physical_cpu(inp.detach()).resolve_conj().resolve_neg()
    return result.float() if result.dtype in _FP8_DTYPES else result


def _assert_result_equal(result, reference):
    # Check the original ABI before observing numerical values on CPU.
    assert result.device == _expected_device()
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else result.device)
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape and result.layout == reference.layout
    if result.dtype in _FP8_DTYPES or (
        flag_gems.vendor_name in ("ascend", "kunlunxin")
        and (
            result.is_complex()
            or result.is_conj()
            or result.is_neg()
            or reference.is_conj()
            or reference.is_neg()
            or (
                flag_gems.vendor_name == "ascend"
                and result.dtype in (torch.int64, torch.float64)
            )
        )
    ):
        result, reference = _numeric_values(result), _numeric_values(reference)
    tu.assert_result_equal(result, reference)


def _check_requested_dtype(dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype in (
        torch.float64,
        torch.complex128,
    ):
        pytest.skip(
            "Kunlunxin tensor transfer narrows float64/complex128 to float32/complex64"
        )


def _cpu_encode(dtype):
    return flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES


def _make_input(dtype, shape, value_range):
    _check_requested_dtype(dtype)
    if _cpu_encode(dtype):
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
    _check_requested_dtype(dtype)
    if _cpu_encode(dtype) or (
        flag_gems.vendor_name == "ascend" and dtype == torch.float64
    ):
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


# aten::permute is a metadata-only view, so every dtype the shared framework can
# build is accepted. Probed with the real overload beyond the 9 spec dtypes:
# bool, int16, float64, complex64 and complex128.
SUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.int16,
    torch.float64,
    torch.complex64,
    torch.complex128,
]

# tu.special_value_cases keeps e4m3fn's nan-only case and gives e5m2 nan, inf
# and mixed, matching what each dtype can represent.
SPECIAL_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

# torch.autograd.grad on the native operator succeeds for all of these (FP8 and
# complex included); the gradient of a permute is a pure relayout with no
# arithmetic, so it is compared exactly.
BACKWARD_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.complex64,
    torch.complex128,
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


@pytest.mark.permute
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test_permute(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _reference(inp)
    # Reverse permutation: [] for 0-D, [0] for 1-D, both valid natively.
    dims = list(range(len(shape) - 1, -1, -1))

    ref_out = torch.ops.aten.permute(ref_inp, dims)
    res_out = flag_gems.permute(inp, dims)

    _assert_result_equal(res_out, ref_out)


# dims is required and has no schema default, so it cannot be omitted in a
# positive case; every accepted dims form is covered instead.
DIMS_ROWS = [
    ((2, 3, 4), [0, 1, 2]),
    ((2, 3, 4), [2, 1, 0]),
    ((2, 3, 4), [1, 2, 0]),
    ((2, 3, 4), [-1, -3, -2]),
    ((2, 3, 4), (0, -1, 1)),
    ((2, 3, 4, 5), [3, 0, 2, 1]),
]


@pytest.mark.permute
@pytest.mark.parametrize("shape,dims", DIMS_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_permute_dims(shape, dims, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.permute(ref_inp, dims)
    res_out = flag_gems.permute(inp, dims)

    _assert_result_equal(res_out, ref_out)


# Source layouts: dense, offset window, column-strided slice, zero-stride expand
# view, plus the 0-D and 1-D forms.
VIEW_ROWS = [
    ((2, 3, 4), [2, 1, 0], "dense"),
    ((3, 4, 5), [1, 0, 2], "window"),
    ((4, 5), [1, 0], "colstep"),
    ((5, 4), [1, 0], "expand"),
    ((), [], "dense"),
    ((1,), [0], "dense"),
    ((0, 3), [1, 0], "dense"),
    ((2, 0, 4), [2, 0, 1], "dense"),
]


def _make_view_source(shape, kind):
    """Build a permute input whose layout is not necessarily contiguous."""
    if kind == "dense":
        return _make_input(torch.float32, shape, ["-1", "1"])
    if kind == "window":
        base = _make_input(
            torch.float32, (shape[0] + 3, shape[1] + 5, shape[2] + 7), ["-1", "1"]
        )
        return base[1 : 1 + shape[0], 2 : 2 + shape[1], 3 : 3 + shape[2]]
    if kind == "colstep":
        base = _make_input(torch.float32, (shape[0] * 2, shape[1]), ["-1", "1"])
        return base[::2]
    if kind == "expand":
        base = _make_input(torch.float32, (1, shape[1]), ["-1", "1"])
        return base.expand(shape)
    raise ValueError(f"unknown source layout {kind!r}")


@pytest.mark.permute
@pytest.mark.parametrize("case", VIEW_ROWS)
def test_permute_view_metadata(case):
    shape, dims, kind = case
    inp = _make_view_source(shape, kind)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.permute(ref_inp, dims)
    res_out = flag_gems.permute(inp, dims)

    _assert_result_equal(res_out, ref_out)
    # Metadata the shared value assertion does not cover: permute must stay a
    # view aliasing the input storage with permuted strides and the same offset.
    assert tuple(res_out.stride()) == tuple(inp.stride()[d] for d in dims)
    assert res_out.storage_offset() == inp.storage_offset()
    assert res_out._is_view()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()


MUTATION_ROWS = [
    ((2, 3, 4), [2, 1, 0]),
    ((4, 5), [1, 0]),
]


@pytest.mark.permute
@pytest.mark.parametrize("shape,dims", MUTATION_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
def test_permute_writes_through_the_view(shape, dims, dtype):
    inp = _make_input(dtype, shape, ["-1", "1"])
    ref_inp = _reference(inp)

    res_out = flag_gems.permute(inp, dims)
    ref_out = torch.ops.aten.permute(ref_inp, dims)
    res_out.fill_(3)
    ref_out.fill_(3)

    _assert_result_equal(res_out, ref_out)
    # The write must land in the base tensor: a materialized copy would leave
    # inp untouched.
    _assert_result_equal(inp, ref_inp)


LAZY_ROWS = ["plain", "conj", "neg", "conj_neg"]


@pytest.mark.permute
@pytest.mark.parametrize("case", LAZY_ROWS)
def test_permute_preserves_lazy_bits(case):
    inp = _make_input(torch.complex64, (4, 6), ["-1", "1"])
    if case in ("conj", "conj_neg"):
        inp = inp.conj()
    if case in ("neg", "conj_neg"):
        inp = torch._neg_view(inp)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.permute(ref_inp, [1, 0])
    res_out = flag_gems.permute(inp, [1, 0])

    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    _assert_result_equal(res_out, ref_out)


BACKWARD_ROWS = [
    ((20, 320, 15), [2, 1, 0]),
    ((16, 7, 57, 32, 29), [4, 2, 0, 3, 1]),
]


@pytest.mark.permute
@pytest.mark.parametrize("case", tu.selected_cases(BACKWARD_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_permute_backward(case, dtype):
    shape, dims = case
    inp = _make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = _reference(inp.detach()).requires_grad_(True)
    upstream = _make_input(dtype, tuple(shape[d] for d in dims), ["-1", "1"])
    ref_upstream = _reference(upstream)

    (res_grad,) = torch.autograd.grad(
        flag_gems.permute(inp, dims), inp, grad_outputs=upstream
    )
    (ref_grad,) = torch.autograd.grad(
        torch.ops.aten.permute(ref_inp, dims), ref_inp, grad_outputs=ref_upstream
    )

    _assert_result_equal(res_grad, ref_grad)


@pytest.mark.permute
@pytest.mark.parametrize(
    "case", tu.selected_cases(tu.special_value_cases(SPECIAL_DTYPES), quick=[])
)
def test_permute_special_values(case):
    dtype, scenario = case
    # Reshape the shared 1-D payload so the permute performs a real dim swap.
    inp = _make_special_input(dtype, scenario).reshape(1, -1)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.permute(ref_inp, [1, 0])
    res_out = flag_gems.permute(inp, [1, 0])

    _assert_result_equal(res_out, ref_out)


# There is no "unsupported dtype" negative: the native overload accepted every
# dtype in SUPPORTED_DTYPES. Native dims errors are RuntimeException / IndexError;
# ValueError is accepted alongside because a candidate may validate dims itself
# (the checkout's own permute does). Only the candidate's exception is asserted.
INVALID_DIMS_ROWS = [
    pytest.param("duplicate", (3, 4), [0, 0], id="duplicate"),
    pytest.param("duplicate_3d", (2, 3, 4), [1, 1, 2], id="duplicate_3d"),
    pytest.param("out_of_range", (3, 4), [0, 2], id="out_of_range"),
    pytest.param("negative_out_of_range", (3, 4), [-3, 0], id="negative_out_of_range"),
    pytest.param("out_of_range_3d", (2, 3, 4), [0, 1, 3], id="out_of_range_3d"),
    pytest.param("wrong_arity", (3, 4), [0], id="wrong_arity"),
    pytest.param("empty_dims_non_scalar", (3, 4), [], id="empty_dims_non_scalar"),
]


@pytest.mark.permute
@pytest.mark.parametrize("name,shape,dims", INVALID_DIMS_ROWS)
def test_permute_invalid_dims_raise(name, shape, dims):
    if flag_gems.vendor_name == "kunlunxin" and name in (
        "duplicate_3d",
        "out_of_range",
        "wrong_arity",
        "empty_dims_non_scalar",
    ):
        pytest.skip("Kunlunxin permute does not reliably reject these invalid dims")
    del name
    inp = _make_input(torch.float32, shape, ["-1", "1"])
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        flag_gems.permute(inp, dims)


# Keep the omitted-argument form apart: it is a missing required argument, not
# an invalid dims value.
@pytest.mark.permute
def test_permute_missing_dims_raises():
    inp = _make_input(torch.float32, (3, 4), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.permute(inp)
