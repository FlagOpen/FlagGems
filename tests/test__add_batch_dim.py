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
from torch._C._functorch import is_legacy_batchedtensor

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


def _local_comparison_to_cpu(tensor):
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


def _local_comparison_values(result, reference):
    from . import accuracy_utils as utils

    assert result.dtype == reference.dtype
    if result.dtype in tu._FP8_DTYPES or (
        flag_gems.vendor_name in ("ascend", "kunlunxin") and result.is_complex()
    ):
        assert result.shape == reference.shape
        assert result.layout == reference.layout
        if utils.TO_CPU:
            assert reference.device == torch.device("cpu")
        else:
            assert result.device == reference.device
        return (_local_comparison_to_cpu(result), _local_comparison_to_cpu(reference))
    return (utils.to_cpu(result, reference), reference)


def _local_assert_result_equal(result, reference):
    result, reference = _local_comparison_values(result, reference)
    torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)


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


# _add_batch_dim hides a physical dimension in a legacy BatchedTensor view.
# Unwrapping with the same level must recover the original values and layout.
# Autograd rejects legacy BatchedTensor outputs, so there are no backward cases.
_ADD_BATCH_DIM_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool, torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
)
# Scalars have no dimension to hide and are covered by a rejection test.
_ADD_BATCH_DIM_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 1]
_ADD_BATCH_DIM_CASES = [
    (shape, batch_dim)
    for shape in _ADD_BATCH_DIM_SHAPES
    for batch_dim in sorted({0, len(shape) // 2, len(shape) - 1})
]


def _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level):
    # A plain tensor can pass an unwrap round-trip for singleton batch dims.
    # Require the actual legacy batch wrapper as well as its visible metadata.
    assert is_legacy_batchedtensor(res_out)
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()

    # Unwrapping must recover the original physical input exactly.
    res_mat = torch.ops.aten._remove_batch_dim(
        res_out, level, ref_inp.size(batch_dim), batch_dim
    )
    _local_assert_result_equal(res_mat, ref_inp)
    assert torch._C._is_alias_of(res_mat, inp)


@pytest.mark.underscore_add_batch_dim
@pytest.mark.parametrize("shape, batch_dim", _ADD_BATCH_DIM_CASES)
@pytest.mark.parametrize("level", [0, 1, 3])
@pytest.mark.parametrize("dtype", _ADD_BATCH_DIM_DTYPES)
def test__add_batch_dim(shape, batch_dim, level, dtype):
    inp = _local_make_input(dtype, shape, ["-1", "1"])
    ref_inp = _local_to_reference(inp)

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = flag_gems._add_batch_dim(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _ADD_BATCH_DIM_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.underscore_add_batch_dim
@pytest.mark.parametrize("shape", _ADD_BATCH_DIM_SHAPES)
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test__add_batch_dim_value_ranges(shape, dtype, value_range):
    batch_dim = len(shape) // 2
    level = 0
    inp = _local_make_input(dtype, shape, value_range)
    ref_inp = _local_to_reference(inp)

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = flag_gems._add_batch_dim(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level)


@pytest.mark.underscore_add_batch_dim
@pytest.mark.parametrize("shape, batch_dim", [((8, 16, 32), 1), ((4, 8, 16, 32), 2)])
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", _ADD_BATCH_DIM_DTYPES)
def test__add_batch_dim_non_contiguous(shape, batch_dim, level, dtype):
    base = _local_make_input(dtype, shape, ["-1", "1"])
    ref_base = _local_to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = flag_gems._add_batch_dim(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level)


@pytest.mark.underscore_add_batch_dim
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_ADD_BATCH_DIM_DTYPES))
)
def test__add_batch_dim_nan_inf(dtype, scenario):
    inp = _local_make_special_input(dtype, scenario)
    ref_inp = _local_to_reference(inp)
    batch_dim, level = 0, 0

    ref_out = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    res_out = flag_gems._add_batch_dim(inp, batch_dim, level)

    _assert_batched_view(res_out, ref_out, inp, ref_inp, batch_dim, level)


@pytest.mark.underscore_add_batch_dim
def test__add_batch_dim_rejects_0dim_input():
    inp = _local_make_input(torch.float32, (), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._add_batch_dim(inp, 0, 0)
    with pytest.raises(RuntimeError):
        flag_gems._add_batch_dim(inp, 0, 0)


@pytest.mark.underscore_add_batch_dim
def test__add_batch_dim_rejects_negative_level():
    inp = _local_make_input(torch.float32, (4, 5), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._add_batch_dim(inp, 1, -1)
    with pytest.raises(RuntimeError):
        flag_gems._add_batch_dim(inp, 1, -1)


@pytest.mark.underscore_add_batch_dim
def test__add_batch_dim_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._add_batch_dim(3.14, 0, 0)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._add_batch_dim(3.14, 0, 0)
