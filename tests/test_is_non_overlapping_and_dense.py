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


# aten::is_non_overlapping_and_dense(Tensor self) -> bool is decided entirely by
# sizes / strides / storage_offset, so it has no value semantics, never
# broadcasts an operand and is not differentiable (autograd.grad on its bool
# result raises TypeError: 'bool' object is not iterable).  Native probes on the
# active backend returned a Python bool for every dtype below, False for a sparse
# COO self and RuntimeError for a non-Tensor self.

# Static runtime capability flags, read once at import time (no probe call).
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    return flag_name is None or bool(getattr(flag_gems.runtime.device, flag_name))


DTYPES = [
    dtype
    for dtype in (
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
        torch.complex128,
    )
    if _dtype_supported(dtype)
]


def _assert_bool_result(result, reference):
    # The schema returns a Python bool: compare type and value directly.
    assert isinstance(result, bool)
    assert result == reference


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test_is_non_overlapping_and_dense(dtype, shape, value_range):
    inp = _local_make_input(dtype, shape, value_range)
    ref_inp = _local_to_reference(inp)

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_inp)
    res_out = flag_gems.is_non_overlapping_and_dense(inp)

    _assert_bool_result(res_out, ref_out)


_BASE_SHAPE = (4, 6)

# Layouts of one 4x6 storage: contiguity, dense permutations (via view and via
# as_strided strides), strided and offset views, singleton / empty / scalar
# boundaries, stride-0 expansion and as_strided gaps / overlaps.  These rows are
# the whole semantics of the predicate and each allocates one small storage, so
# every row runs in quick mode too.
_LAYOUTS = {
    "contiguous": lambda t: t,
    "transposed": lambda t: t.t(),
    "permuted": lambda t: t.reshape(2, 2, 6).permute(2, 0, 1),
    "strided_last_dim": lambda t: t[:, ::2],
    "strided_3d_last_dim": lambda t: t.reshape(2, 2, 6)[..., ::2],
    "sliced_transposed": lambda t: t[:, 1:5].t(),
    "expanded_rows": lambda t: t[:1].expand(4, 6),
    "expanded_3d": lambda t: t[:1].reshape(1, 2, 3).expand(5, 2, 3),
    "offset_view": lambda t: t.reshape(-1)[6:].reshape(3, 6),
    "narrow": lambda t: t[1:3],
    "singleton_slice": lambda t: t[:1],
    "select_index": lambda t: t[0],
    "scalar_view": lambda t: t[0, 0],
    "empty_view": lambda t: t[:0],
    "as_strided_gap": lambda t: t.as_strided((2, 3), (7, 1)),
    "as_strided_overlap": lambda t: t.as_strided((2, 4), (1, 1)),
    "as_strided_dense_perm": lambda t: t.as_strided((3, 2, 4), (1, 12, 3)),
}

_LAYOUT_DTYPES = [
    torch.float32,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.bool,
    torch.complex64,
]

_LAYOUT_ROWS = [
    (layout, dtype)
    for layout in _LAYOUTS
    for dtype in _LAYOUT_DTYPES
    if _dtype_supported(dtype)
]


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("layout,dtype", _LAYOUT_ROWS)
def test_is_non_overlapping_and_dense_layout(layout, dtype):
    base = _local_make_input(dtype, _BASE_SHAPE, ["-1", "1"])
    inp = _LAYOUTS[layout](base)
    meta = (tuple(inp.shape), inp.stride(), inp.storage_offset())
    ref_inp = _local_to_reference(inp)

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_inp)
    res_out = flag_gems.is_non_overlapping_and_dense(inp)

    # The predicate must not disturb the input's own layout metadata.
    assert (tuple(inp.shape), inp.stride(), inp.storage_offset()) == meta
    _assert_bool_result(res_out, ref_out)


@pytest.mark.is_non_overlapping_and_dense
def test_is_non_overlapping_and_dense_sparse():
    # A COO tensor is not a dense layout: the native predicate returns False
    # instead of raising, so the candidate must agree on that input form.
    def _sparse():
        indices = torch.tensor([[0, 1], [1, 2]], device=flag_gems.device)
        values = torch.ones(2, dtype=torch.float32, device=flag_gems.device)
        return torch.sparse_coo_tensor(indices, values, (4, 6))

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(_sparse())
    res_out = flag_gems.is_non_overlapping_and_dense(_sparse())

    _assert_bool_result(res_out, ref_out)


_SPECIAL_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

# nan / inf / mixed payloads for every supported floating dtype.  The predicate
# reads metadata only, so each payload is also observed through a stride-0
# expanded view: a candidate that inspected element values instead of strides
# would diverge there.  Default-only, like the other positive special cases.
_SPECIAL_ROWS = tu.selected_cases(
    [
        (dtype, scenario, arrangement)
        for dtype, scenario in tu.special_value_cases(
            [dtype for dtype in _SPECIAL_DTYPES if _dtype_supported(dtype)]
        )
        for arrangement in ("dense", "expanded")
    ],
    quick=[],
)


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("dtype,scenario,arrangement", _SPECIAL_ROWS)
def test_is_non_overlapping_and_dense_special_values(dtype, scenario, arrangement):
    base = _local_make_special_input(dtype, scenario)
    inp = base if arrangement == "dense" else base[:1].expand(base.shape)
    ref_inp = _local_to_reference(inp)

    ref_out = torch.ops.aten.is_non_overlapping_and_dense(ref_inp)
    res_out = flag_gems.is_non_overlapping_and_dense(inp)

    _assert_bool_result(res_out, ref_out)


# The native schema rejects a non-Tensor 'self' with RuntimeError and a call
# without arguments with TypeError; both levels keep every row.
_NEGATIVE_ARGS = [3.14, [1.0, 2.0], "tensor"]


@pytest.mark.is_non_overlapping_and_dense
@pytest.mark.parametrize("arg", _NEGATIVE_ARGS)
def test_is_non_overlapping_and_dense_invalid_argument(arg):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_non_overlapping_and_dense(arg)


@pytest.mark.is_non_overlapping_and_dense
def test_is_non_overlapping_and_dense_missing_argument():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_non_overlapping_and_dense()
