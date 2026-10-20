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
    if flag_gems.vendor_name == "ascend" and dtype in (
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
    cpu_encoding = (
        flag_gems.vendor_name == "ascend"
        and dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    ) or (flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype == torch.complex32)
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


def _assert_payload_equal(result, reference):
    # Validate the selected reference ABI before observing values on CPU.
    assert result.dtype == reference.dtype and result.shape == reference.shape
    assert result.layout == reference.layout
    expected = _device("cpu" if utils.TO_CPU else flag_gems.device)
    assert result.device == reference.device == expected
    torch.testing.assert_close(
        _copy_to(result, "cpu"),
        _copy_to(reference, "cpu"),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


# aten::is_conj(Tensor self) -> bool only reads the tensor's lazy conjugate bit,
# so coverage varies that state (and the dtype, because only complex tensors can
# carry the bit) instead of comparing computed values. Broadcast and backward do
# not apply: there is no second operand and no differentiable output.

_COMPLEX_DTYPES = [torch.complex32, torch.complex64] + (
    [torch.complex128] if utils.fp64_is_supported else []
)

_DTYPE_FLAGS = {
    torch.bfloat16: "bf16_is_supported",
    torch.float8_e4m3fn: "fp8_is_supported",
    torch.float8_e5m2: "fp8_is_supported",
    torch.float64: "fp64_is_supported",
}


def _dtype_supported(dtype):
    flag = _DTYPE_FLAGS.get(dtype)
    return getattr(utils, flag) if flag else True


# Probed on the active backend: every REQUIRED_DTYPES entry, float64, int16 and
# bool are accepted at all five spec ranges.
_DTYPES = (
    [d for d in tu.REQUIRED_DTYPES if _dtype_supported(d)]
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.int16, torch.bool]
    + _COMPLEX_DTYPES
)


def _conj_state(base, state):
    """Return a tensor in the requested lazy-flag state, preserving layout."""
    if (
        flag_gems.vendor_name == "ascend"
        and base.dtype == torch.complex32
        and state in ("conj_resolved", "conj_cloned")
    ):
        cpu = _copy_to(base, "cpu").conj()
        cpu = cpu.resolve_conj() if state == "conj_resolved" else cpu.clone()
        return _copy_to(cpu, base.device)
    if state == "plain":
        return base
    if state == "conj":
        return base.conj()
    if state == "conj_transposed":
        return base.conj().transpose(0, 1)
    if state == "transposed_conj":
        return base.transpose(0, 1).conj()
    if state == "conj_sliced":
        return base.conj()[1:-1]
    if state == "conj_strided":
        return base.conj().transpose(1, 2)
    if state == "conj_narrowed":
        return base.conj().narrow(0, 1, base.shape[0] - 2)
    if state == "conj_expanded":
        # expand only broadcasts size-1 dimensions, so insert one first: the
        # leading stride becomes 0 while the conjugate bit and inner strides stay.
        return base.conj().unsqueeze(0).expand(2, *base.shape)
    if state == "conj_axis_swapped":
        return base.conj().transpose(0, -1)
    if state == "conj_resolved":
        return base.conj().resolve_conj()
    if state == "conj_cloned":
        return base.conj().clone()
    if state == "conj_contiguous":
        return base.conj().contiguous()
    if state == "conj_detached":
        return base.conj().detach()
    if state == "neg":
        return torch._neg_view(base)
    if state == "neg_of_conj":
        return torch._neg_view(base.conj())
    if state == "conj_of_neg":
        return torch._neg_view(base).conj()
    if state == "neg_of_conj_transposed":
        return torch._neg_view(base.conj()).transpose(0, 1)
    raise AssertionError(f"unknown state {state}")


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


@pytest.mark.is_conj
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,value_range",
    [pytest.param(dtype, value_range) for dtype, value_range in _DTYPE_RANGE_CASES],
)
def test_is_conj(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _reference(inp)

    ref_out = torch.ops.aten.is_conj(ref_inp)
    res_out = flag_gems.is_conj(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    # Querying must not set the flag it reads.
    assert inp.is_conj() == ref_inp.is_conj()


# State rows: each row keeps a distinct lazy-flag composition or view layout.
_STATE_ROWS = [
    ("plain", (1024, 1024)),
    ("conj", (1024, 1024)),
    ("conj_transposed", (20, 320, 15)),
    ("transposed_conj", (20, 320, 15)),
    ("conj_sliced", (20, 320, 15)),
    ("conj_strided", (16, 128, 64, 60)),
    ("conj_narrowed", (1024, 1024)),
    ("conj_expanded", (16, 128, 64, 60)),
    ("conj_axis_swapped", (16, 7, 57, 32, 29)),
    ("conj_resolved", (1024, 1024)),
    ("conj_cloned", (1024, 1024)),
    ("conj_contiguous", (16, 128, 64, 60)),
    ("conj_detached", (1024, 1024)),
    ("neg", (1024, 1024)),
    ("neg_of_conj", (20, 320, 15)),
    ("conj_of_neg", (20, 320, 15)),
    ("neg_of_conj_transposed", (16, 128, 64, 60)),
]

# Quick keeps every lazy-flag state on the smallest shapes the compositions
# accept, so no state is dropped: slice and narrow need dim0 >= 3 to stay
# non-empty, everything else fits the quick shape.
_QUICK_STATE_ROWS = [
    (state, (4, 19, 7) if state in ("conj_sliced", "conj_narrowed") else (2, 19, 7))
    for state, _ in _STATE_ROWS
]

_STATE_CASES = tu.selected_cases(
    _STATE_ROWS + _QUICK_STATE_ROWS, quick=_QUICK_STATE_ROWS
)


@pytest.mark.is_conj
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(dtype) for dtype in _COMPLEX_DTYPES + [torch.float32]],
)
@pytest.mark.parametrize("state,shape", _STATE_CASES)
def test_is_conj_state(state, shape, dtype):
    base = _make_input(dtype, shape, tu.selected_ranges()[0])
    ref_base = _reference(base)

    inp = _conj_state(base, state)
    ref_inp = _conj_state(ref_base, state)

    ref_out = torch.ops.aten.is_conj(ref_inp)
    res_out = flag_gems.is_conj(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert inp.is_conj() == ref_inp.is_conj()
    assert inp.is_neg() == ref_inp.is_neg()


_BOUNDARY_DTYPES = _COMPLEX_DTYPES + [torch.float32, torch.int64, torch.bool]


@pytest.mark.is_conj
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(dtype) for dtype in _BOUNDARY_DTYPES],
)
@pytest.mark.parametrize("state", ["plain", "conj"])
@pytest.mark.parametrize("shape", [(), (1,), (0,), (3, 0)])
def test_is_conj_boundary_shapes(shape, state, dtype):
    base = _make_input(dtype, shape, tu.selected_ranges()[0])
    ref_base = _reference(base)

    inp = _conj_state(base, state)
    ref_inp = _conj_state(ref_base, state)

    ref_out = torch.ops.aten.is_conj(ref_inp)
    res_out = flag_gems.is_conj(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


# Complex special values are added explicitly; the shared generator already
# exempts e4m3fn inf/mixed because fp8 cannot represent inf.
_SPECIAL_DTYPES = [d for d in _DTYPES if d.is_floating_point]
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(_SPECIAL_DTYPES)
    + [(d, s) for d in _COMPLEX_DTYPES for s in ("nan", "inf", "mixed")],
    quick=[],
)


@pytest.mark.is_conj
@pytest.mark.parametrize(
    "dtype,scenario",
    [pytest.param(dtype, scenario) for dtype, scenario in _SPECIAL_CASES],
)
@pytest.mark.parametrize("state", ["plain", "conj"])
def test_is_conj_special_values(state, dtype, scenario):
    base = _make_special_input(dtype, scenario).to(flag_gems.device)
    ref_base = _reference(base)

    inp = _conj_state(base, state)
    ref_inp = _conj_state(ref_base, state)

    ref_out = torch.ops.aten.is_conj(ref_inp)
    res_out = flag_gems.is_conj(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


@pytest.mark.is_conj
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(dtype) for dtype in _COMPLEX_DTYPES + [torch.float32]],
)
@pytest.mark.parametrize("shape", [(256,), (20, 320, 15)])
def test_is_conj_query_leaves_input_untouched(shape, dtype):
    # Querying a lazy-conj view must neither resolve the flag nor touch the
    # storage it aliases.
    inp = _conj_state(_make_input(dtype, shape, tu.selected_ranges()[0]), "conj")
    # to_reference already re-applies the lazy conjugate bit, so a second .conj()
    # would toggle it back off and compare against a plain tensor.
    ref_inp = _reference(inp)

    expected_device, expected_dtype = inp.device, inp.dtype
    before_ptr = inp.data_ptr()
    before_offset = inp.storage_offset()
    before_stride = inp.stride()
    before_storage = (
        inp.untyped_storage().data_ptr(),
        inp.untyped_storage().nbytes(),
    )
    before_state = (inp.is_conj(), inp.is_neg(), inp.shape)
    before_payload = _reference(inp)

    ref_out = torch.ops.aten.is_conj(ref_inp)
    res_out = flag_gems.is_conj(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert inp.device == expected_device and inp.dtype == expected_dtype
    assert inp.data_ptr() == before_ptr
    assert inp.storage_offset() == before_offset
    assert inp.stride() == before_stride
    assert (inp.untyped_storage().data_ptr(), inp.untyped_storage().nbytes()) == (
        before_storage
    )
    assert (inp.is_conj(), inp.is_neg(), inp.shape) == before_state
    _assert_payload_equal(_reference(inp), before_payload)


@pytest.mark.is_conj
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(dtype) for dtype in _COMPLEX_DTYPES],
)
@pytest.mark.parametrize("shape", [(1024, 1024)])
def test_is_conj_flag_is_per_tensor_not_per_storage(shape, dtype):
    base = _make_input(dtype, shape, tu.selected_ranges()[0])
    ref_base = _reference(base)

    # A lazy-conj alias and a plain alias of the same storage.
    inp_view = base.conj()
    ref_view = ref_base.conj()
    inp_alias = base.view(shape)
    ref_alias = ref_base.view(shape)

    assert inp_view.data_ptr() == base.data_ptr()
    assert ref_view.data_ptr() == ref_base.data_ptr()

    view_ref = torch.ops.aten.is_conj(ref_view)
    view_res = flag_gems.is_conj(inp_view)
    alias_ref = torch.ops.aten.is_conj(ref_alias)
    alias_res = flag_gems.is_conj(inp_alias)

    assert type(view_res) is bool
    assert type(alias_res) is bool
    assert view_res == view_ref
    assert alias_res == alias_ref
    assert view_res != alias_res


# Negative cases stay in every mode. None is not a row: native
# aten::is_conj(None) binds the optional Tensor argument and returns False, so
# it is a valid call rather than invalid input.
_NON_TENSOR_ARGS = [3.14, 7, "abc", [1.0, 2.0], (1, 2)]


@pytest.mark.is_conj
@pytest.mark.parametrize("bad_arg", _NON_TENSOR_ARGS)
def test_is_conj_rejects_non_tensor_args(bad_arg):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.is_conj(bad_arg)


@pytest.mark.is_conj
def test_is_conj_requires_an_argument():
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.is_conj()


@pytest.mark.is_conj
def test_is_conj_undefined_tensor():
    ref = torch.ops.aten.is_conj(None)
    res = flag_gems.is_conj(None)

    assert isinstance(res, bool)
    assert res is ref
