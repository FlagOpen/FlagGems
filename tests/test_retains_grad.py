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

"""Correctness tests for ``aten::retains_grad``.

``retains_grad(Tensor self) -> bool`` reports Autograd state: it answers True
only for a non-leaf tensor whose own node called ``retain_grad()``. The query
reads no element data, so the five value ranges act as dtype/storage forms of a
leaf tensor, the nan/inf matrix checks value acceptance, and broadcast plus
tensor-vs-scalar do not apply to a single-operand schema. Backward is covered
as the post-backward state, because the operator has no differentiable output.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Backend compatibility stays local to this operator suite.


def _local_make_input(dtype, shape, value_range):
    """Build a tensor of ``shape`` / ``dtype`` with values in ``value_range``.

    ``value_range`` is a [low_symbol, high_symbol] pair whose symbols resolve
    per-dtype (max/min are the dtype bounds). bool ignores the range; integer
    ranges are snapped to ints; a degenerate range (low == high) fills the
    constant; everything else uses torch.testing.make_tensor (complex fills
    both real and imaginary parts).

    Bounds are clamped to the dtype's representable range first, so the spec's
    five ranges work unchanged for dtypes that cannot represent a bound (e.g.
    uint8 cannot hold ``-1``, so ``[-1,0]`` becomes the degenerate ``[0,0]``
    → a constant zero fill). This keeps the caller from having to special-case
    unsigned dtypes.
    """
    low = tu.resolve_bound(value_range[0], dtype)
    high = tu.resolve_bound(value_range[1], dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()
    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = (int(low), int(high))
        dtype_min, dtype_max = tu.dtype_bounds(dtype)
        low = max(low, int(dtype_min))
        high = min(high, int(dtype_max))
        low = min(low, high)
    make_fp8_on_cpu = flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    creation_device = "cpu" if make_fp8_on_cpu else flag_gems.device
    if low == high:
        result = torch.full(shape, low, device=creation_device, dtype=dtype)
    else:
        result = torch.testing.make_tensor(
            shape, dtype=dtype, device=creation_device, low=low, high=high
        )
    return result.to(flag_gems.device) if make_fp8_on_cpu else result


def _local_to_reference(inp, upcast=False):
    """Independent tensor oracle for these tests, retaining view metadata.

    A plain clone or device transfer can compact strides, reset offsets or
    materialize lazy flags. Copy dense storage to the reference device before
    rebuilding the view. Optional upcasting still follows accuracy_utils' value
    conversion policy. This does not reproduce inter-input aliases or autograd
    history; version-counter tests use their own reference setup.
    """
    from . import accuracy_utils as utils

    if inp is None:
        return None
    if inp.layout != torch.strided or inp.is_quantized or inp.is_nested:
        reference = inp.detach().clone()
    else:
        storage = inp.untyped_storage()
        if utils.TO_CPU and inp.device.type != "cpu":
            storage = storage.cpu()
        else:
            storage = storage.clone()
        reference = torch.empty(0, dtype=inp.dtype, device=storage.device).set_(
            storage, inp.storage_offset(), inp.size(), inp.stride()
        )
        if inp.is_conj():
            reference = reference.conj()
        if inp.is_neg():
            reference = torch._neg_view(reference)
    reference.requires_grad_(inp.requires_grad)
    return utils.to_reference(reference, upcast)


def _local_make_special_input(dtype, scenario):
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    make_fp8_on_cpu = flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    creation_device = "cpu" if make_fp8_on_cpu else flag_gems.device
    result = torch.tensor(
        payloads[scenario], device=creation_device, dtype=torch.float32
    ).to(dtype)
    return result.to(flag_gems.device) if make_fp8_on_cpu else result


_DTYPE_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.complex128: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}

# Every dtype the schema accepts: the nine required ones plus bool and
# complex64, and float64 where the device supports it. A tensor of an integer
# or bool dtype is a valid operand; it simply cannot carry autograd state, so
# those dtypes are covered by the storage/mutation grids rather than by the
# state grid.
_VALUE_DTYPES = list(tu.REQUIRED_DTYPES) + [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _VALUE_DTYPES.append(torch.float64)
_VALUE_DTYPES = [dtype for dtype in _VALUE_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

_STATE_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.complex64,
]
if utils.fp64_is_supported:
    _STATE_DTYPES.append(torch.float64)
_STATE_DTYPES = [dtype for dtype in _STATE_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

_LEAF = "leaf"
_LEAF_RETAINED = "leaf_retain_grad"
_NONLEAF = "nonleaf"
_NONLEAF_RETAINED = "nonleaf_retained"
_NONLEAF_DEEP = "nonleaf_deep_retained"
_CLONE_OF_RETAINED = "clone_of_retained"
_RETAINED_VIEW = "retained_view"
_VIEW_OF_RETAINED = "view_of_retained"
_DETACHED = "detached"
_NO_GRAD = "no_grad"
_INFERENCE_MODE = "inference_mode"
_PARAMETER = "parameter"

_STATE_SCENARIOS = [
    _LEAF,
    _LEAF_RETAINED,
    _NONLEAF,
    _NONLEAF_RETAINED,
    _NONLEAF_DEEP,
    _CLONE_OF_RETAINED,
    _RETAINED_VIEW,
    _VIEW_OF_RETAINED,
    _DETACHED,
    _NO_GRAD,
    _INFERENCE_MODE,
    _PARAMETER,
]

# Only these states answer True: the tensor is a non-leaf and retain_grad() was
# called on that same node.
_RETAINED_SCENARIOS = frozenset({_NONLEAF_RETAINED, _NONLEAF_DEEP, _RETAINED_VIEW})


def _swap_last_two(node):
    # t() is defined only for dim <= 2, and the state grid reaches 5-D.
    return node.t() if node.dim() <= 2 else node.transpose(-1, -2)


def _state_fixture(scenario, dtype, shape):
    """Build a tensor in ``scenario``'s Autograd state.

    The query reads no element, so the fixture stays unfilled. Each call builds
    its own graph, giving candidate and reference the same state without
    sharing a node: ``tu.to_reference`` copies storage into a fresh leaf, which
    answers False for every retained non-leaf. ``clone()`` is the
    graph-building op because it is differentiable for every dtype here,
    including both FP8 types.
    """
    base = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    base = base.requires_grad_(True)
    if scenario == _LEAF:
        return base
    if scenario == _LEAF_RETAINED:
        # retain_grad() is a documented no-op on a leaf, which already keeps
        # its .grad, so the answer stays False.
        base.retain_grad()
        return base
    if scenario == _NONLEAF:
        return base.clone()
    if scenario == _NONLEAF_RETAINED:
        node = base.clone()
        node.retain_grad()
        return node
    if scenario == _NONLEAF_DEEP:
        node = base.clone().clone()
        node.retain_grad()
        return node
    if scenario == _CLONE_OF_RETAINED:
        node = base.clone()
        node.retain_grad()
        return node.clone()
    if scenario == _RETAINED_VIEW:
        node = _swap_last_two(base.clone())
        node.retain_grad()
        return node
    if scenario == _VIEW_OF_RETAINED:
        # A view of a retained node is a new node: the flag is not inherited.
        node = base.clone()
        node.retain_grad()
        return _swap_last_two(node)
    if scenario == _DETACHED:
        return base.clone().detach()
    if scenario == _NO_GRAD:
        with torch.no_grad():
            return base.clone()
    if scenario == _INFERENCE_MODE:
        with torch.inference_mode():
            return base.clone()
    if scenario == _PARAMETER:
        return torch.nn.Parameter(base)
    raise ValueError("unknown state scenario: %r" % (scenario,))


def _autograd_snapshot(node):
    """Autograd/storage metadata that a state query must leave untouched."""
    return (
        node.data_ptr(),
        node.storage_offset(),
        tuple(node.stride()),
        node.requires_grad,
        node.is_leaf,
        node.grad_fn is not None,
    )


# Explicit default/quick pairs; dtype lists retain device capability filtering.
_SHAPE_DTYPE_CASES = tu.selected_cases(
    [
        ((), torch.float32),
        ((1,), torch.float32),
        ((256,), torch.float32),
        ((1024, 1024), torch.float32),
        ((16, 128, 64, 60), torch.float32),
        ((16, 7, 57, 32, 29), torch.float32),
    ]
    + [((20, 320, 15), dtype) for dtype in _VALUE_DTYPES],
    quick=[((2, 19, 7), dtype) for dtype in _VALUE_DTYPES],
)


@pytest.mark.retains_grad
@pytest.mark.parametrize("shape, dtype", _SHAPE_DTYPE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges()[:1])
def test_retains_grad_plain_tensor(shape, value_range, dtype):
    # A tensor that no operation produced is a leaf, so every supported dtype
    # and storage form (0-dim, strided, bool, FP8, complex) answers False.
    inp = _local_make_input(dtype, shape, value_range)
    ref_inp = _local_to_reference(inp)

    ref_out = torch.ops.aten.retains_grad(ref_inp)
    res_out = flag_gems.retains_grad(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert res_out is False


@pytest.mark.retains_grad
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _STATE_DTYPES)
@pytest.mark.parametrize("scenario", _STATE_SCENARIOS)
def test_retains_grad_autograd_state(shape, dtype, scenario):
    ref_inp = _state_fixture(scenario, dtype, shape)
    inp = _state_fixture(scenario, dtype, shape)

    ref_out = torch.ops.aten.retains_grad(ref_inp)
    res_out = flag_gems.retains_grad(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert res_out is (scenario in _RETAINED_SCENARIOS)


_VIEW_LAYOUTS = [
    ("transposed", True),
    ("transposed", False),
    ("strided_slice", True),
    ("strided_slice", False),
    ("offset_window", True),
    ("offset_window", False),
]


def _apply_view(node, layout):
    # A 2-D base keeps t() valid; every layout is non-contiguous, and the
    # window additionally lands on a nonzero storage offset (row 2, column 3).
    if layout == "transposed":
        return node.t()
    if layout == "strided_slice":
        return node[:, ::2]
    if layout == "offset_window":
        return node[2:6, 3:9]
    raise ValueError("unknown view layout: %r" % (layout,))


def _view_fixture(layout, retained, dtype):
    base = torch.empty((8, 12), dtype=dtype, device=flag_gems.device)
    inp = _apply_view(base.requires_grad_(True).clone(), layout)
    if retained:
        inp.retain_grad()
    return inp


@pytest.mark.retains_grad
@pytest.mark.parametrize("layout,retained", _VIEW_LAYOUTS)
@pytest.mark.parametrize("dtype", _STATE_DTYPES)
def test_retains_grad_view_layout(layout, retained, dtype):
    ref_inp = _view_fixture(layout, retained, dtype)
    inp = _view_fixture(layout, retained, dtype)
    before = _autograd_snapshot(inp)

    ref_out = torch.ops.aten.retains_grad(ref_inp)
    res_out = flag_gems.retains_grad(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert res_out is retained
    # Snapshot and re-query are both read on the candidate input after the
    # call: a query must not re-view, reallocate or flag this node.
    assert _autograd_snapshot(inp) == before
    assert torch.ops.aten.retains_grad(inp) is retained


_CONJ_CASES = ["conj_view_retained", "conj_of_retained"]


def _conj_fixture(case):
    # complex64 only: conj() returns a new node carrying the lazy conjugate bit
    # for complex dtypes, but the very same object for real ones.
    base = torch.empty((4, 6), dtype=torch.complex64, device=flag_gems.device)
    if case == "conj_view_retained":
        node = base.requires_grad_(True).clone().conj()
        node.retain_grad()
        return node
    node = base.requires_grad_(True).clone()
    node.retain_grad()
    return node.conj()


@pytest.mark.retains_grad
@pytest.mark.parametrize("case", _CONJ_CASES)
def test_retains_grad_lazy_conj_view(case):
    ref_inp = _conj_fixture(case)
    inp = _conj_fixture(case)
    before = _autograd_snapshot(inp)

    ref_out = torch.ops.aten.retains_grad(ref_inp)
    res_out = flag_gems.retains_grad(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert res_out is (case == "conj_view_retained")
    # The lazy conjugate bit is only queried, never materialized.
    assert inp.is_conj() == ref_inp.is_conj()
    # Read on the candidate input after the call.
    assert _autograd_snapshot(inp) == before
    assert torch.ops.aten.retains_grad(inp) is (case == "conj_view_retained")


@pytest.mark.retains_grad
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test_retains_grad_does_not_mutate_input(shape, dtype):
    # A leaf accepts every dtype, so the candidate is also checked on the
    # integer/bool storage forms it must accept but never re-flag.
    ref_inp = _local_make_input(dtype, shape, ["-1", "1"])
    inp = _local_make_input(dtype, shape, ["-1", "1"])
    before = _autograd_snapshot(inp)

    ref_out = torch.ops.aten.retains_grad(ref_inp)
    res_out = flag_gems.retains_grad(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert res_out is False
    # Reallocating or re-viewing would change the snapshot, and retaining this
    # node would flip the next query to True.
    assert _autograd_snapshot(inp) == before
    assert torch.ops.aten.retains_grad(inp) is False


@pytest.mark.retains_grad
@pytest.mark.parametrize("dtype", _STATE_DTYPES)
def test_retains_grad_does_not_mutate_retained_nonleaf(dtype):
    # A retained non-leaf is the state the candidate could damage most easily.
    ref_inp = _state_fixture(_NONLEAF_RETAINED, dtype, (8, 12))
    inp = _state_fixture(_NONLEAF_RETAINED, dtype, (8, 12))
    before = _autograd_snapshot(inp)

    ref_out = torch.ops.aten.retains_grad(ref_inp)
    res_out = flag_gems.retains_grad(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert res_out is True
    assert _autograd_snapshot(inp) == before
    assert torch.ops.aten.retains_grad(inp) is True


_BACKWARD_STATES = tu.selected_cases(["retained", "not_retained"], quick=[])
_BACKWARD_DTYPES = [
    dtype
    for dtype in (torch.float16, torch.float32, torch.bfloat16)
    if _DTYPE_FLAGS.get(dtype, True)
]
if utils.fp64_is_supported:
    _BACKWARD_DTYPES.append(torch.float64)

_BACKWARD_SHAPES = [(8, 12), (4, 5, 6)]


def _backward_fixture(state, dtype, shape):
    # The operator has no differentiable output of its own, so the backward
    # pass is only run to completion; real values keep it meaningful.
    base = _local_make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    out = base * 3
    if state == "retained":
        out.retain_grad()
    (out * out).sum().backward()
    return out


@pytest.mark.retains_grad
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
@pytest.mark.parametrize("state", _BACKWARD_STATES)
def test_retains_grad_after_backward(shape, dtype, state):
    # The flag must survive the completed backward pass, which is what
    # retain_grad() promises: the retained node keeps its .grad.
    retained = state == "retained"
    ref_inp = _backward_fixture(state, dtype, shape)
    inp = _backward_fixture(state, dtype, shape)

    ref_out = torch.ops.aten.retains_grad(ref_inp)
    res_out = flag_gems.retains_grad(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert res_out is retained
    if retained:
        assert inp.grad is not None


_SPECIAL_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
if utils.fp64_is_supported:
    _SPECIAL_DTYPES.append(torch.float64)
_SPECIAL_DTYPES = [dtype for dtype in _SPECIAL_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# e4m3fn contributes the nan scenario only, per the shared generator's dtype
# table: that dtype cannot represent infinity. Every remaining (dtype,
# scenario) row is checked on a leaf and on a retained non-leaf, which clone()
# also builds for FP8.
_SPECIAL_ROWS = [
    (dtype, scenario, state)
    for dtype, scenario in tu.special_value_cases(_SPECIAL_DTYPES)
    for state in ("leaf", "retained_nonleaf")
]


@pytest.mark.retains_grad
@pytest.mark.parametrize(
    "dtype,scenario,state", tu.selected_cases(_SPECIAL_ROWS, quick=[])
)
def test_retains_grad_special_values(dtype, scenario, state):
    def build():
        node = _local_make_special_input(dtype, scenario).requires_grad_(True)
        if state == "retained_nonleaf":
            node = node.clone()
            node.retain_grad()
        return node

    ref_inp = build()
    inp = build()

    ref_out = torch.ops.aten.retains_grad(ref_inp)
    res_out = flag_gems.retains_grad(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert res_out is (state == "retained_nonleaf")


# The schema has no dtype/dim restriction, no parameters and no scalar-operand
# overload, so the only rejected arguments are a non-tensor operand, a missing
# and an extra operand. NaN/Inf tensors are valid inputs, covered above.
_REJECTED_ARGUMENTS = [3.14, 1, True, "tensor", [1, 2], (1, 2), [[0.0]], None]


@pytest.mark.retains_grad
@pytest.mark.parametrize("value", _REJECTED_ARGUMENTS)
def test_retains_grad_rejects_non_tensor(value):
    with pytest.raises((TypeError, RuntimeError, ValueError)):
        flag_gems.retains_grad(value)


@pytest.mark.retains_grad
@pytest.mark.parametrize("wrong_arity", ["missing_operand", "extra_operand"])
def test_retains_grad_rejects_wrong_arity(wrong_arity):
    inp = _local_make_input(torch.float32, (4,), ["-1", "1"])
    args = (inp, inp) if wrong_arity == "extra_operand" else ()
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.retains_grad(*args)
