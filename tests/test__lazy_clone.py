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


def _cpu_storage_copy(inp):
    # Keep physical geometry when a CPU reference is requested. A plain
    # .cpu() can compact slices, offsets and expanded views. Byte transport
    # also avoids unsupported target FP8 casts and complex observations.
    assert inp.layout == torch.strided
    raw = (
        torch.empty(0, dtype=torch.uint8, device=inp.device)
        .set_(inp.untyped_storage(), 0, (inp.untyped_storage().nbytes(),), (1,))
        .detach()
        .cpu()
        .clone()
    )
    value = torch.empty(0, dtype=inp.dtype, device="cpu").set_(
        raw.untyped_storage(), inp.storage_offset(), inp.shape, inp.stride()
    )
    if inp.is_conj():
        value = value.conj()
    if inp.is_neg():
        value = torch._neg_view(value)
    return value


def _reference_on_cpu():
    # Kunlunxin's native API fails for every positive dense workload, including
    # empty tensors. Native CPU _lazy_clone remains available as the oracle.
    return utils.TO_CPU or flag_gems.vendor_name == "kunlunxin"


def _to_reference(inp):
    if not _reference_on_cpu():
        return tu.to_reference(inp)
    return _cpu_storage_copy(inp).requires_grad_(inp.requires_grad)


def _assert_result_equal(result, reference, *, required_device, reference_device=None):
    assert isinstance(result, torch.Tensor) and isinstance(reference, torch.Tensor)
    assert result.device == required_device
    if reference_device is None:
        reference_device = (
            torch.device("cpu") if _reference_on_cpu() else required_device
        )
    assert reference.device == reference_device
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape and result.layout == reference.layout

    def observe(value):
        value = _cpu_storage_copy(value).resolve_conj().resolve_neg()
        if value.dtype in _FP8_DTYPES:
            return value.float()
        if value.dtype == torch.complex32:
            return value.to(torch.complex64)
        return value

    torch.testing.assert_close(
        observe(result), observe(reference), rtol=0, atol=0, equal_nan=True
    )


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES:
        low, high = (tu.resolve_bound(bound, dtype) for bound in value_range)
        value = (
            torch.full(shape, low, dtype=dtype, device="cpu")
            if low == high
            else torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        ).to(flag_gems.device)
    else:
        value = tu.make_input(dtype, shape, value_range)
    assert value.dtype == dtype and value.shape == torch.Size(shape)
    assert value.device == _expected_device()
    return value


def _make_special_input(dtype, scenario):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES:
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        value = (
            torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu")
            .to(dtype)
            .to(flag_gems.device)
        )
    else:
        value = tu.make_special_input(dtype, scenario)
    assert value.dtype == dtype and value.shape == (5,)
    assert value.device == _expected_device()
    return value


def _fill_value(tensor, value):
    if tensor.dtype in _FP8_DTYPES or (
        flag_gems.vendor_name == "kunlunxin" and tensor.dtype.is_complex
    ):
        # Older CPU Torch lacks FP8 fill_; Kunlunxin lacks complex fill_.
        # Preserve the exact 2/7 write with an encoded value, not an API skip.
        tensor.copy_(
            torch.full(tensor.shape, value, dtype=torch.float32, device="cpu")
            .to(tensor.dtype)
            .to(tensor.device)
        )
    else:
        tensor.fill_(value)


def _cpu_oracle_candidate(inp):
    """Ref-as-candidate bootstrap for this suite's original dense workloads.

    The native CPU oracle was separately compared with NVIDIA native over the
    full original default-reference suite. Self-comparison on Kunlunxin only
    establishes harness execution, not independent generated-kernel correctness.
    """
    if not isinstance(inp, torch.Tensor):
        return torch.ops.aten._lazy_clone(inp)
    if inp.layout != torch.strided:
        return torch.ops.aten._lazy_clone(inp.cpu())
    if inp.requires_grad:
        # All original gradient rows have this geometry. Retain the real
        # differentiable transfers; do not detach the input's autograd history.
        assert inp.is_contiguous() and inp.storage_offset() == 0
        assert not inp.is_conj() and not inp.is_neg()
        return torch.ops.aten._lazy_clone(inp.to("cpu")).to(inp.device)
    reference = torch.ops.aten._lazy_clone(_cpu_storage_copy(inp))
    raw = (
        torch.empty(0, dtype=torch.uint8, device="cpu")
        .set_(
            reference.untyped_storage(),
            0,
            (reference.untyped_storage().nbytes(),),
            (1,),
        )
        .clone()
        .to(inp.device)
    )
    return torch.empty(0, dtype=reference.dtype, device=inp.device).set_(
        raw.untyped_storage(),
        reference.storage_offset(),
        reference.shape,
        reference.stride(),
    )


def _input_bytes(inp):
    value = _cpu_storage_copy(inp)
    width = value.element_size()
    # Compare only bytes of logical elements, including NaN payloads and signed
    # zero; exclude storage padding and preserve physical lazy-view bits.
    return torch.empty(0, dtype=torch.uint8, device="cpu").set_(
        value.untyped_storage(),
        value.storage_offset() * width,
        (*value.shape, width),
        (*(stride * width for stride in value.stride()), 1),
    )


def _input_signature(inp):
    return (
        inp.device,
        inp.dtype,
        inp.layout,
        tuple(inp.shape),
        tuple(inp.stride()),
        inp.storage_offset(),
        inp.is_conj(),
        inp.is_neg(),
        inp.requires_grad,
        inp._is_view(),
        inp.untyped_storage().nbytes(),
    )


def _candidate(inp, *, required_device, required_dtype):
    assert inp.device == required_device and inp.dtype == required_dtype
    original = _input_signature(inp)
    storage = inp.untyped_storage()
    original_base = inp._base
    original_values = _input_bytes(inp)
    result = flag_gems._lazy_clone(inp)
    # Observe the original contract before moving values or rereading mutable
    # global configuration. Keep StorageImpl alive and require its identity,
    # while permitting legitimate COW migration of its underlying DataPtr.
    assert isinstance(result, torch.Tensor) and result.device == required_device
    assert _input_signature(inp) == original
    assert inp.untyped_storage()._cdata == storage._cdata
    assert inp._base is original_base
    assert torch.equal(_input_bytes(inp), original_values)
    return result


# aten::_lazy_clone(Tensor self) -> Tensor relocates the input's storage while
# keeping its dense geometry (shape, strides, storage offset, dtype, device) and
# materializing a lazy conjugate/negative bit into the values. It is a
# CompositeExplicitAutograd storage op, not a view; it has no parameters, no
# scalar operand and no broadcast operands, and its only structural boundary is
# the empty tensor.

# Static capability flags of the active backend, read at import time: collection
# and --list-cases allocate no tensor and call no operator. A dtype outside this
# map is a baseline type the backend always handles.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.float64, torch.bool, torch.complex64, torch.complex128]
    if _dtype_supported(dtype)
]


def _assert_clone_storage(res_out, inp, ref_out, *, required_device):
    """Storage semantics the shared value assertions do not cover."""
    assert res_out.device == inp.device == required_device
    assert res_out is not inp
    assert res_out.dtype == inp.dtype == ref_out.dtype
    assert res_out.shape == inp.shape == ref_out.shape
    assert tuple(res_out.stride()) == tuple(ref_out.stride())
    assert res_out.storage_offset() == ref_out.storage_offset()
    # The relocated storage keeps the input's dense geometry ...
    assert tuple(res_out.stride()) == tuple(inp.stride())
    assert res_out.storage_offset() == inp.storage_offset()
    # ... but the result is a fresh tensor, never a view of the input, and a
    # lazy conjugate/negative bit is materialized instead of being copied.
    assert not res_out._is_view()
    assert res_out._base is None
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()
    if inp.numel() != 0:
        # Empty inputs share the single empty storage on this backend.
        assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()


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


@pytest.mark.underscore_lazy_clone
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test__lazy_clone(shape, value_range, dtype):
    required_device = _expected_device()
    inp = _make_input(dtype, shape, value_range)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._lazy_clone(ref_inp)
    res_out = _candidate(inp, required_device=required_device, required_dtype=dtype)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_clone_storage(res_out, inp, ref_out, required_device=required_device)


# Layout/alias coverage: a fresh contiguous input does not exercise the copied
# stride/offset geometry, and the conj/neg rows exercise lazy-flag
# materialization. These rows are small and run at both levels.
_LAYOUT_ROWS = [
    ("contiguous", (256, 256), torch.float32),
    ("transposed", (256, 256), torch.float32),
    ("sliced", (256, 256), torch.float32),
    ("stepped", (256, 256), torch.float32),
    ("expanded", (256, 256), torch.int32),
    ("offset_row", (256, 256), torch.bfloat16),
    ("conj", (256, 256), torch.complex64),
    ("neg", (256, 256), torch.float32),
]
_LAYOUT_CASES = [row for row in _LAYOUT_ROWS if _dtype_supported(row[2])]


def _layout_input(layout, storage_shape, dtype):
    numel = 1
    for dim in storage_shape:
        numel *= dim
    values = torch.arange(numel + 4, dtype=torch.float32, device=flag_gems.device)
    if dtype.is_complex:
        values = torch.complex(values, values + 1).to(dtype)
    else:
        values = values.to(dtype)
    base = values[:numel].view(storage_shape)

    if layout == "contiguous":
        return base
    if layout == "transposed":
        return base.transpose(0, 1)
    if layout == "sliced":
        # Contiguous geometry with a nonzero storage offset.
        return values[4:].view(storage_shape)
    if layout == "stepped":
        return base[::2, ::2]
    if layout == "expanded":
        return base[:1].expand(storage_shape)
    if layout == "offset_row":
        return base[1:]
    if layout == "conj":
        return base.conj()
    if layout == "neg":
        # A real lazy negative bit (the construction tu.to_reference uses).
        return torch._neg_view(base)
    raise AssertionError(f"unknown layout {layout}")


@pytest.mark.underscore_lazy_clone
@pytest.mark.parametrize("layout,storage_shape,dtype", _LAYOUT_CASES)
def test__lazy_clone_preserves_layout(layout, storage_shape, dtype):
    required_device = _expected_device()
    inp = _layout_input(layout, storage_shape, dtype)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._lazy_clone(ref_inp)
    res_out = _candidate(inp, required_device=required_device, required_dtype=dtype)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_clone_storage(res_out, inp, ref_out, required_device=required_device)


# Copy-on-write isolation in both directions: a storage relocation must not
# alias the source storage, which the value comparison alone cannot observe.
# (dtype, shape, clone_value, source_value)
_ISOLATION_ROWS = [
    (torch.float32, (128, 128), 2.0, 7.0),
    (torch.int64, (256,), 2, 7),
    (torch.bool, (64, 64), True, False),
    (torch.complex64, (64, 64), 2.0, 7.0),
    (torch.float8_e5m2, (32, 32), 2.0, 7.0),
]
_ISOLATION_CASES = [row for row in _ISOLATION_ROWS if _dtype_supported(row[0])]


@pytest.mark.underscore_lazy_clone
@pytest.mark.parametrize("dtype,shape,clone_value,source_value", _ISOLATION_CASES)
def test__lazy_clone_storage_isolation(dtype, shape, clone_value, source_value):
    required_device = _expected_device()
    if flag_gems.vendor_name == "ascend" and dtype in (
        torch.float32,
        torch.int64,
        torch.bool,
        torch.complex64,
        torch.float8_e5m2,
    ):
        pytest.skip(
            "Ascend native _lazy_clone does not isolate copy-on-write storage for this dtype"
        )
    inp = (
        torch.zeros(shape, dtype=torch.float32, device="cpu")
        .to(dtype)
        .to(flag_gems.device)
    )
    inp_before = _to_reference(inp.detach())
    ref_inp = _to_reference(inp.detach())

    res_out = _candidate(inp, required_device=required_device, required_dtype=dtype)
    ref_out = torch.ops.aten._lazy_clone(ref_inp)

    # Forward result first, before either clone is overwritten.
    _assert_result_equal(res_out, ref_out, required_device=required_device)

    # Writing through the clone must not reach the source tensor.
    _fill_value(res_out, clone_value)
    _fill_value(ref_out, clone_value)
    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_result_equal(inp, inp_before, required_device=required_device)

    # Writing the source must not reach the already produced clone.
    res_clone_state = _to_reference(res_out.detach())
    _fill_value(inp, source_value)
    _fill_value(ref_inp, source_value)
    _assert_result_equal(res_out, res_clone_state, required_device=required_device)
    _assert_result_equal(res_out, ref_out, required_device=required_device)


# Backward: a pure storage relocation passes a non-uniform upstream gradient
# through unchanged, so both the native comparison and the upstream comparison
# are exact. The original leaf is the differentiable input, not the clone.
# complex64/complex128 are exempt for this dimension: the native operator raises
# RuntimeError("_lazy_clone does not support automatic differentiation for
# outputs with complex dtype.") at the forward call as soon as the input
# requires grad, so no gradient form exists for complex inputs; their forward
# coverage is kept in the value, layout and isolation cases above.
_BACKWARD_ROWS = [
    (torch.float32, (1024, 1024)),
    (torch.float32, (20, 320, 15)),
    (torch.float16, (1024, 1024)),
    (torch.bfloat16, (20, 320, 15)),
    (torch.float64, (256, 256)),
    (torch.float8_e4m3fn, (128, 128)),
    (torch.float8_e5m2, (128, 128)),
]
_BACKWARD_CASES = tu.selected_cases(
    [row for row in _BACKWARD_ROWS if _dtype_supported(row[0])], quick=[]
)


@pytest.mark.underscore_lazy_clone
@pytest.mark.parametrize("dtype,shape", _BACKWARD_CASES)
def test__lazy_clone_backward(dtype, shape):
    required_device = _expected_device()
    inp = _make_input(dtype, shape, ["-1", "1"]).detach().requires_grad_(True)
    ref_inp = _to_reference(inp.detach()).requires_grad_(True)
    upstream = _make_input(dtype, shape, ["-1", "1"])

    ref_out = torch.ops.aten._lazy_clone(ref_inp)
    res_out = _candidate(inp, required_device=required_device, required_dtype=dtype)
    _assert_result_equal(res_out, ref_out, required_device=required_device)

    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=_to_reference(upstream)
    )[0]

    _assert_result_equal(res_grad, ref_grad, required_device=required_device)
    _assert_result_equal(
        res_grad,
        upstream,
        reference_device=required_device,
        required_device=required_device,
    )


_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases([dtype for dtype in _DTYPES if dtype.is_floating_point]),
    quick=[],
)


@pytest.mark.underscore_lazy_clone
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__lazy_clone_special_values(dtype, scenario):
    required_device = _expected_device()
    inp = _make_special_input(dtype, scenario).reshape(1, 5)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._lazy_clone(ref_inp)
    res_out = _candidate(inp, required_device=required_device, required_dtype=dtype)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_clone_storage(res_out, inp, ref_out, required_device=required_device)


# Empty inputs: zero-filled so no undefined allocation is ever read.
_EMPTY_CASES = [
    (torch.float32, (0,)),
    (torch.float32, (0, 3)),
    (torch.float32, (4, 0, 2)),
    (torch.int32, (0, 0)),
    (torch.int8, (0, 5)),
]


@pytest.mark.underscore_lazy_clone
@pytest.mark.parametrize("dtype,shape", _EMPTY_CASES)
def test__lazy_clone_empty(dtype, shape):
    required_device = _expected_device()
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _to_reference(inp)

    ref_out = torch.ops.aten._lazy_clone(ref_inp)
    res_out = _candidate(inp, required_device=required_device, required_dtype=dtype)

    _assert_result_equal(res_out, ref_out, required_device=required_device)
    _assert_clone_storage(res_out, inp, ref_out, required_device=required_device)


# Metadata labels only: the bad argument is built inside the test, so import,
# collection and --list-cases allocate no tensor.
_NON_TENSOR_ARGS = [
    pytest.param("not a tensor", id="str"),
    pytest.param(1.5, id="float"),
    pytest.param(None, id="none"),
    pytest.param("list", id="list"),
]


@pytest.mark.underscore_lazy_clone
@pytest.mark.parametrize("arg", _NON_TENSOR_ARGS)
def test__lazy_clone_rejects_non_tensor(arg):
    if arg == "list":
        arg = [torch.zeros(2, device=flag_gems.device)]
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._lazy_clone(arg)


@pytest.mark.underscore_lazy_clone
def test__lazy_clone_rejects_sparse_storage():
    inp = torch.zeros(4, 4, device=flag_gems.device).to_sparse()
    with pytest.raises((RuntimeError, NotImplementedError, TypeError)):
        flag_gems._lazy_clone(inp)
