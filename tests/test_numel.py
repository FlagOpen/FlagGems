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

import math

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


def _check_input(inp, dtype, shape):
    assert inp.dtype == dtype and inp.shape == torch.Size(shape)
    assert inp.device == _device(flag_gems.device)
    return inp


def _cpu_input(dtype, shape, value_range):
    # The original factory recipe, with the same bounds and degenerate fills.
    low, high = (tu.resolve_bound(v, dtype) for v in value_range)
    if not dtype.is_floating_point:
        low, high = int(low), int(high)
        dtype_min, dtype_max = tu.dtype_bounds(dtype)
        low = max(low, int(dtype_min))
        high = min(high, int(dtype_max))
        low = min(low, high)
    if low == high:
        return torch.full(shape, low, dtype=dtype, device="cpu")
    return torch.testing.make_tensor(
        shape, dtype=dtype, device="cpu", low=low, high=high
    )


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        inp = _copy_to(_cpu_input(dtype, shape, value_range), flag_gems.device)
    else:
        inp = tu.make_input(dtype, shape, value_range)
    return _check_input(inp, dtype, shape)


def _make_special_input(dtype, scenario):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
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


def _repeat(inp, *sizes):
    if (
        flag_gems.vendor_name in ("ascend", "kunlunxin")
        and inp.dtype
        in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        and inp.device.type != "cpu"
    ):
        return _copy_to(_copy_to(inp, "cpu").repeat(*sizes), inp.device)
    return inp.repeat(*sizes)


def _zeros(shape, *, dtype, device):
    if (
        flag_gems.vendor_name in ("ascend", "kunlunxin")
        and dtype
        in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        and torch.device(device).type != "cpu"
    ):
        inp = _copy_to(torch.zeros(shape, dtype=dtype, device="cpu"), device)
    else:
        inp = torch.zeros(shape, dtype=dtype, device=device)
    assert inp.dtype == dtype and inp.shape == torch.Size(shape)
    assert inp.device == _device(device)
    return inp


# aten::numel(Tensor self) -> int is a host-side metadata query: it has a single
# ``default`` overload (no ``.out``), one tensor operand (no broadcast) and an
# int result (no backward), so those spec dimensions do not apply. Shape,
# layout, offset and allocation-state semantics are covered instead; the value
# range never changes the result but keeps the shared grid.
_NUMEL_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.int16, torch.bool, torch.complex64]
)
_NUMEL_SPECIAL_DTYPES = [dtype for dtype in _NUMEL_DTYPES if dtype.is_floating_point]

# Strided states that separate the logical size from the storage layout.
_STRIDED_LAYOUT_KINDS = [
    "dense",
    "transpose",
    "step_slice",
    "narrowed",
    "offset_view",
    "zero_stride",
    "empty_1d",
    "empty_3d",
    "scalar",
    "conj_view",
]

_STRUCTURED_LAYOUT_KINDS = [
    "sparse_coo",
    "sparse_coo_hybrid",
    "sparse_coo_uncoalesced",
    "sparse_csr",
    "nested_strided",
    "nested_jagged",
    "quantized_per_tensor",
    "quantized_per_channel",
    "meta",
]

# None is not listed: aten::numel(None) returns 0 instead of raising, so it is a
# positive case below rather than a rejected argument.
_NON_TENSOR_ARG_CASES = [
    pytest.param(3.14, id="float"),
    pytest.param(1, id="int"),
    pytest.param(True, id="bool"),
    pytest.param("abc", id="str"),
    pytest.param((1, 2), id="tuple"),
    pytest.param([1, 2], id="list"),
    pytest.param({"a": 1}, id="dict"),
    pytest.param(object(), id="object"),
]


def _strided_tensor(kind, dtype, device):
    if kind == "dense":
        return _zeros((4, 6), dtype=dtype, device=device)
    if kind == "transpose":
        return _zeros((6, 4), dtype=dtype, device=device).t()
    if kind == "step_slice":
        return _zeros((8, 8), dtype=dtype, device=device)[::2]
    if kind == "narrowed":
        return _zeros((8, 8), dtype=dtype, device=device).narrow(0, 2, 3)
    if kind == "offset_view":
        base = _zeros((6, 6), dtype=dtype, device=device)
        return torch.empty(0, dtype=dtype, device=device).set_(
            base.untyped_storage(), 10, (5,), (2,)
        )
    if kind == "zero_stride":
        return _zeros((4, 1), dtype=dtype, device=device).expand(4, 6)
    if kind == "empty_1d":
        return torch.empty(0, dtype=dtype, device=device)
    if kind == "empty_3d":
        return torch.empty((0, 3, 4), dtype=dtype, device=device)
    if kind == "scalar":
        return _zeros((), dtype=dtype, device=device)
    if kind == "conj_view":
        return _zeros((4, 4), dtype=dtype, device=device).conj()
    raise ValueError(f"Unknown strided layout kind: {kind!r}")


def _quantize(kind, device):
    source = torch.ones((4, 5), device=device)
    if kind == "quantized_per_tensor":
        return torch.quantize_per_tensor(source, 0.1, 3, torch.quint8)
    return torch.quantize_per_channel(
        source,
        torch.tensor([0.1, 0.2, 0.3, 0.4], device=device),
        torch.tensor([3, 4, 5, 6], device=device),
        0,
        torch.quint8,
    )


def _structured_tensor(kind, device):
    if kind == "sparse_coo":
        indices = torch.tensor([[0, 1, 2], [1, 0, 3]], device=device)
        values = torch.ones(3, device=device)
        return torch.sparse_coo_tensor(indices, values, (4, 6), device=device)
    if kind == "sparse_coo_hybrid":
        indices = torch.tensor([[0, 1], [1, 0]], device=device)
        values = torch.ones(2, 2, device=device)
        return torch.sparse_coo_tensor(indices, values, (4, 6, 2), device=device)
    if kind == "sparse_coo_uncoalesced":
        indices = torch.tensor([[0, 0], [1, 1]], device=device)
        values = torch.ones(2, device=device)
        return torch.sparse_coo_tensor(indices, values, (4, 6), device=device)
    if kind == "sparse_csr":
        crow = torch.tensor([0, 1, 2, 3], device=device)
        col = torch.tensor([0, 1, 2], device=device)
        values = torch.ones(3, device=device)
        return torch.sparse_csr_tensor(crow, col, values, size=(3, 6), device=device)
    if kind == "nested_strided":
        parts = [
            torch.ones(2, 3, device=device),
            torch.ones(5, 3, device=device),
        ]
        return torch.nested.nested_tensor(parts)
    if kind == "nested_jagged":
        parts = [
            torch.ones(2, 3, device=device),
            torch.ones(5, 3, device=device),
        ]
        return torch.nested.nested_tensor(parts, layout=torch.jagged)
    if kind in ("quantized_per_tensor", "quantized_per_channel"):
        return _quantize(kind, device)
    if kind == "meta":
        return torch.zeros(3, 4, dtype=torch.float32, device="meta")
    raise ValueError(f"Unknown structured layout kind: {kind!r}")


def _strided_state(tensor):
    """Host-side allocation metadata that a metadata query must not touch."""
    return (
        tuple(tensor.size()),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.data_ptr(),
        tensor.untyped_storage()._cdata,
        tensor.dtype,
        tensor.device,
        tensor.untyped_storage().nbytes(),
        tensor._version,
    )


def _special_tensor(shape, dtype, scenario):
    count = math.prod(shape)
    values = _make_special_input(dtype, scenario)
    repeats = (count + values.numel() - 1) // values.numel()
    return _repeat(values, repeats)[:count].reshape(shape)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _NUMEL_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test_numel_value_ranges(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.numel(ref_inp)
    res_out = flag_gems.numel(inp)

    assert type(res_out) is int
    assert res_out == ref_out


@pytest.mark.numel
@pytest.mark.parametrize("kind", _STRIDED_LAYOUT_KINDS)
@pytest.mark.parametrize("dtype", _NUMEL_DTYPES)
def test_numel_strided_layouts(kind, dtype):
    inp = _strided_tensor(kind, dtype, flag_gems.device)
    # The reference is rebuilt on the reference device: moving a view would
    # compact its storage and change the offset/stride being tested.
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _strided_tensor(kind, dtype, ref_device)
    before = _strided_state(inp)

    ref_out = torch.ops.aten.numel(ref_inp)
    res_out = flag_gems.numel(inp)

    assert type(res_out) is int
    assert res_out == ref_out
    # A shape-only query must leave storage, stride and offset untouched.
    assert _strided_state(inp) == before


@pytest.mark.numel
@pytest.mark.parametrize("kind", _STRUCTURED_LAYOUT_KINDS)
def test_numel_structured_layouts(kind):
    inp = _structured_tensor(kind, flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _structured_tensor(kind, ref_device)
    version_before = inp._version

    ref_out = torch.ops.aten.numel(ref_inp)
    res_out = flag_gems.numel(inp)

    assert type(res_out) is int
    assert res_out == ref_out
    assert inp._version == version_before
    if kind.startswith("sparse_coo"):
        # The dense element count, not the number of stored entries.
        assert res_out != inp._nnz()
    if kind == "sparse_csr":
        assert res_out != inp.values().numel()


@pytest.mark.numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_NUMEL_SPECIAL_DTYPES), quick=[]),
)
def test_numel_special_values(shape, dtype, scenario):
    inp = _special_tensor(shape, dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.numel(ref_inp)
    res_out = flag_gems.numel(inp)

    assert type(res_out) is int
    assert res_out == ref_out


@pytest.mark.numel
def test_numel_ignores_autograd():
    inp = torch.zeros((4, 6), dtype=torch.float32, device=flag_gems.device)
    inp.requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    version_before = inp._version

    ref_out = torch.ops.aten.numel(ref_inp)
    res_out = flag_gems.numel(inp)

    assert type(res_out) is int
    assert res_out == ref_out
    # The count carries no grad_fn and does not bump the version counter.
    assert inp._version == version_before
    assert inp.grad is None


@pytest.mark.numel
def test_numel_none_returns_zero():
    # aten::numel(None) is accepted and returns 0, so this is a positive case
    # and None stays out of the rejected-argument list.
    ref_out = torch.ops.aten.numel(None)
    res_out = flag_gems.numel(None)

    assert type(res_out) is int
    assert res_out == ref_out


@pytest.mark.numel
@pytest.mark.parametrize("arg", _NON_TENSOR_ARG_CASES)
def test_numel_rejects_non_tensor(arg):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.numel(arg)


@pytest.mark.numel
def test_numel_rejects_missing_argument():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.numel()
