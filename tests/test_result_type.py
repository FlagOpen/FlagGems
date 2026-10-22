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

"""Correctness tests for aten::result_type.

aten::result_type is pure dtype inference: each of its four overloads
(.Tensor, .Scalar, .Scalar_Tensor, .Scalar_Scalar) returns a Python int
ScalarType code, never a tensor, and no overload reads element values. The
shared shape and value-range grid checks that element values cannot affect the
result. Rank roles also matter: a zero-dimensional tensor can have lower
promotion priority than a dimensioned operand. The tests cover all four call
forms, dtype pairs, rank roles, Python scalar kinds and rejected arguments.
Expectations come directly from the native operator.
"""

import itertools

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}


def _dtype(name):
    dtype = getattr(torch, name, None)
    return dtype if isinstance(dtype, torch.dtype) else None


# fp8 promotes only with a second fp8 operand of the same type: every mixed
# pair raises 'Promotion for Float8 Types is not supported' (probed on the
# active backend), so fp8 dtypes stay out of the general promotion lattice and
# are covered by the dedicated positive and rejected-row tests below.
FP8_DTYPES = [d for d in (_dtype("float8_e4m3fn"), _dtype("float8_e5m2")) if d]

FP8_DTYPES = [dtype for dtype in FP8_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# Pairwise promotion is defined for these twelve types: the integer types cover
# the integer rules and complex64/complex128 the complex rules.
PROMOTION_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
]

PROMOTION_DTYPES = [
    dtype for dtype in PROMOTION_DTYPES if _DTYPE_FLAGS.get(dtype, True)
]

ALL_DTYPES = PROMOTION_DTYPES + FP8_DTYPES


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _device(value):
    device = torch.device(value)
    if device.index is None and device.type not in ("cpu", "meta"):
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return device


def _cpu_copy(value):
    assert value.layout == torch.strided
    raw = (
        torch.empty(0, dtype=torch.uint8, device=value.device)
        .set_(value.untyped_storage(), 0, (value.untyped_storage().nbytes(),), (1,))
        .detach()
        .cpu()
        .clone()
    )
    result = (
        torch.empty(0, dtype=torch.uint8, device="cpu")
        .view(value.dtype)
        .set_(
            raw.untyped_storage(), value.storage_offset(), value.shape, value.stride()
        )
    )
    if value.is_conj():
        result = result.conj()
    if value.is_neg():
        result = torch._neg_view(result)
    return result


def _input_bytes(value):
    copy = _cpu_copy(value)
    width = copy.element_size()
    return torch.empty(0, dtype=torch.uint8, device="cpu").set_(
        copy.untyped_storage(),
        copy.storage_offset() * width,
        (*copy.shape, width),
        (*(v * width for v in copy.stride()), 1),
    )


def _signature(value):
    return (
        type(value),
        value.device,
        value.dtype,
        value.layout,
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        value.is_conj(),
        value.is_neg(),
        value.requires_grad,
        value._is_view(),
        value._version,
        value.untyped_storage().nbytes(),
    )


def _make_input(dtype, shape, value_range):
    device = _device(flag_gems.device)
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES:
        low = tu.resolve_bound(value_range[0], dtype)
        high = tu.resolve_bound(value_range[1], dtype)
        value = (
            torch.full(shape, low, dtype=dtype, device="cpu")
            if low == high
            else torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        ).to(device)
    else:
        value = tu.make_input(dtype, shape, value_range)
    assert value.dtype == dtype and value.device == device
    assert tuple(value.shape) == tuple(shape)
    return value


def _make_special_input(dtype, scenario):
    device = _device(flag_gems.device)
    if (flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES) or (
        flag_gems.vendor_name == "ascend" and dtype == torch.float64
    ):
        payloads = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
        }
        value = (
            torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu")
            .to(dtype)
            .to(device)
        )
    else:
        value = tu.make_special_input(dtype, scenario)
    assert value.dtype == dtype and value.device == device
    assert tuple(value.shape) == (5,)
    return value


def _check_input(value, dtype, shape, device):
    assert value.device == device and value.dtype == dtype
    assert tuple(value.shape) == tuple(shape)
    return value


def _reference(*args):
    values = []
    for value in args:
        if isinstance(value, torch.Tensor):
            expected_device = torch.device("cpu") if utils.TO_CPU else value.device
            reference = _cpu_copy(value) if utils.TO_CPU else tu.to_reference(value)
            _check_input(reference, value.dtype, value.shape, expected_device)
            values.append(reference)
        else:
            values.append(value)
    return torch.ops.aten.result_type(*values)


def _candidate(*args):
    # result_type only observes metadata. Keep both storage identity and every
    # defined element byte so metadata-correct candidates cannot mutate inputs.
    try:
        before = []
        seen = set()
        for value in args:
            if isinstance(value, torch.Tensor) and id(value) not in seen:
                seen.add(id(value))
                assert value.device == _device(flag_gems.device)
                before.append(
                    (
                        value,
                        _signature(value),
                        value.untyped_storage(),
                        value._base,
                        _input_bytes(value),
                    )
                )
    except Exception as error:
        raise AssertionError("result_type input inspection failed") from error
    try:
        return flag_gems.result_type(*args)
    finally:
        try:
            for value, signature, storage, base, payload in before:
                assert _signature(value) == signature
                assert value.untyped_storage()._cdata == storage._cdata
                assert value._base is base
                assert torch.equal(_input_bytes(value), payload)
        except Exception as error:
            # Inspection errors must not satisfy a negative API test.
            raise AssertionError("result_type changed an input") from error


def _make_tensor(dtype, shape):
    # Keep the original zero recipe and rank; encode FP8 on CPU when the
    # backend cannot fill its typed device storage directly.
    device = _device(flag_gems.device)
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES:
        value = torch.zeros(shape, dtype=dtype, device="cpu").to(device)
    else:
        value = torch.zeros(shape, dtype=dtype, device=device)
    return _check_input(value, dtype, shape, device)


def _assert_result(res, ref):
    # result_type returns a Python ScalarType code rather than a tensor, so
    # there is no tensor to hand to the shared value assertions; the schema type
    # is checked and the value is compared against the native oracle.
    assert type(res) is int, f"expected a python int code, got {type(res)}"
    assert res == ref, f"candidate returned {res}, native returned {ref}"


# Every row below is a pair of tiny metadata payloads, so --quick keeps all of
# them; the only rows quick drops are the positive nan/inf payloads and the
# multi-dim rank rows, where the runtime cost actually differs.
TENSOR_PAIR_ROWS = list(itertools.product(PROMOTION_DTYPES, repeat=2))
TENSOR_PAIR_ROWS += [(dtype, dtype) for dtype in FP8_DTYPES]

FP8_MIXED_ROWS = [(fp8, other) for fp8 in FP8_DTYPES for other in PROMOTION_DTYPES]
FP8_MIXED_ROWS += [(other, fp8) for fp8 in FP8_DTYPES for other in PROMOTION_DTYPES]
FP8_MIXED_ROWS += list(itertools.permutations(FP8_DTYPES, 2))

SCALAR_KINDS = [True, 1, 1.5]
COMPLEX_SCALAR = 1 + 2j

TENSOR_SCALAR_ROWS = [(dtype, kind) for dtype in ALL_DTYPES for kind in SCALAR_KINDS]
COMPLEX_SCALAR_ROWS = [(dtype, COMPLEX_SCALAR) for dtype in PROMOTION_DTYPES]
FP8_COMPLEX_SCALAR_ROWS = [(dtype, COMPLEX_SCALAR) for dtype in FP8_DTYPES]

SCALAR_SCALAR_ROWS = list(itertools.product(SCALAR_KINDS + [COMPLEX_SCALAR], repeat=2))

_RANK_SHAPES = {0: (), 1: (4,)}

# The 0-dim tensor boundary against every Python scalar kind, for every dtype.
# These payloads are tiny, so quick keeps all of them.
ZERO_DIM_ROWS = TENSOR_SCALAR_ROWS + COMPLEX_SCALAR_ROWS

# Promotion is dtype driven when both operands share a dtype: a 0-dim, 1-dim,
# 3-dim, 4-dim or 5-dim operand on either side must not change the result.
RANK_ROWS = [
    ((), (), torch.float32),
    ((1,), (1,), torch.float16),
    ((256,), (1024, 1024), torch.float32),
    ((20, 320, 15), (20, 320, 15), torch.bfloat16),
    ((16, 128, 64), (16, 128, 64), torch.float32),
    ((), (2, 3, 4), torch.float32),
    ((1,), (16, 7, 57, 32, 29), torch.float16),
]

# Quick drops only the large-rank rows of this dimension; the small ones stay.
QUICK_RANK_ROWS = [
    row for row in RANK_ROWS if row not in RANK_ROWS[2:5] + RANK_ROWS[6:]
]

# The central mixed-dtype behaviour: with two different dtypes the 0-dim operand
# has a different promotion priority, so exchanging the dtype/rank
# assignments can give different codes (probed on the active backend), e.g.
# uint8[1-D] x int32[0-D] -> uint8 while uint8[0-D] x int32[1-D] -> int32.
MIXED_RANK_ROWS = [
    (torch.uint8, 0, torch.int32, 1),
    (torch.uint8, 1, torch.int32, 0),
    (torch.int8, 0, torch.int32, 1),
    (torch.int8, 1, torch.int32, 0),
    (torch.int32, 0, torch.int64, 1),
    (torch.int32, 1, torch.int64, 0),
    (torch.float16, 0, torch.bfloat16, 1),
    (torch.float16, 1, torch.bfloat16, 0),
    # Two 0-dim operands retain their dtype promotion rules.
    (torch.float16, 0, torch.bfloat16, 0),
]

# nan-only, inf-only and nan+inf payloads for every floating dtype the operator
# accepts; the scenarios the dtype cannot represent are dropped by the shared
# generator itself (e.g. e4m3fn has no infinity).
SPECIAL_VALUE_ROWS = tu.special_value_cases(
    [dtype for dtype in ALL_DTYPES if dtype.is_floating_point]
)

# Objects that match no result_type schema: neither a tensor nor a Scalar.
INVALID_OTHER = ["abc", [1, 2], (1, 2), 2**70]


@pytest.mark.result_type
@pytest.mark.parametrize("a_dtype,b_dtype", TENSOR_PAIR_ROWS)
def test_result_type_tensor_pair(a_dtype, b_dtype):
    a = _make_tensor(a_dtype, (4,))
    b = _make_tensor(b_dtype, (4,))

    ref = _reference(a, b)
    res = _candidate(a, b)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("a_dtype,b_dtype", FP8_MIXED_ROWS)
def test_result_type_fp8_mixed_pair_rejected(a_dtype, b_dtype):
    a = _make_tensor(a_dtype, (4,))
    b = _make_tensor(b_dtype, (4,))

    # An fp8 operand only promotes with a second operand of the same fp8 type.
    with pytest.raises((TypeError, RuntimeError)):
        _candidate(a, b)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", TENSOR_SCALAR_ROWS)
def test_result_type_tensor_scalar(dtype, scalar):
    inp = _make_tensor(dtype, (4,))

    ref = _reference(inp, scalar)
    res = _candidate(inp, scalar)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", TENSOR_SCALAR_ROWS)
def test_result_type_scalar_tensor(dtype, scalar):
    inp = _make_tensor(dtype, (4,))

    ref = _reference(scalar, inp)
    res = _candidate(scalar, inp)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", COMPLEX_SCALAR_ROWS)
def test_result_type_tensor_scalar_complex(dtype, scalar):
    inp = _make_tensor(dtype, (4,))

    ref = _reference(inp, scalar)
    res = _candidate(inp, scalar)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", COMPLEX_SCALAR_ROWS)
def test_result_type_scalar_tensor_complex(dtype, scalar):
    inp = _make_tensor(dtype, (4,))

    ref = _reference(scalar, inp)
    res = _candidate(scalar, inp)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", FP8_COMPLEX_SCALAR_ROWS)
def test_result_type_fp8_complex_scalar_rejected(dtype, scalar):
    inp = _make_tensor(dtype, (4,))

    # A complex Python scalar has no promotion rule for the fp8 types.
    with pytest.raises((TypeError, RuntimeError)):
        _candidate(inp, scalar)


@pytest.mark.result_type
@pytest.mark.parametrize("first,second", SCALAR_SCALAR_ROWS)
def test_result_type_scalar_scalar(first, second):
    ref = _reference(first, second)
    res = _candidate(first, second)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_result_type_tensor_with_none(dtype):
    inp = _make_tensor(dtype, (4,))

    # None matches the optional scalar argument and simply yields the tensor
    # dtype.
    ref = _reference(inp, None)
    res = _candidate(inp, None)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize(
    "shape_a,shape_b,dtype", tu.selected_cases(RANK_ROWS, quick=QUICK_RANK_ROWS)
)
def test_result_type_ignores_tensor_rank(shape_a, shape_b, dtype):
    a = _make_tensor(dtype, shape_a)
    b = _make_tensor(dtype, shape_b)

    ref = _reference(a, b)
    res = _candidate(a, b)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("a_dtype,a_rank,b_dtype,b_rank", MIXED_RANK_ROWS)
def test_result_type_mixed_dtype_rank_role(a_dtype, a_rank, b_dtype, b_rank):
    a = _make_tensor(a_dtype, _RANK_SHAPES[a_rank])
    b = _make_tensor(b_dtype, _RANK_SHAPES[b_rank])

    ref = _reference(a, b)
    res = _candidate(a, b)

    # Swapping dtype/rank assignments can change promotion priority.
    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,scalar", ZERO_DIM_ROWS)
def test_result_type_scalar_over_0d_tensor(dtype, scalar):
    inp = _make_tensor(dtype, ())

    ref = _reference(inp, scalar)
    res = _candidate(inp, scalar)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(SPECIAL_VALUE_ROWS, quick=[])
)
def test_result_type_ignores_element_values(dtype, scenario):
    inp = _make_special_input(dtype, scenario)

    ref = _reference(inp, inp)
    res = _candidate(inp, inp)

    _assert_result(res, ref)


@pytest.mark.result_type
@pytest.mark.parametrize("other", INVALID_OTHER)
def test_result_type_tensor_invalid_other(other):
    inp = _make_tensor(torch.float32, (4,))

    # Newer Torch versions reject the out-of-range Python integer with
    # OverflowError; other invalid schema arguments retain their old contract.
    errors = (
        (TypeError, RuntimeError, OverflowError)
        if type(other) is int
        else (TypeError, RuntimeError)
    )
    with pytest.raises(errors):
        _candidate(inp, other)


@pytest.mark.result_type
@pytest.mark.parametrize("first,second", [("a", "b"), ([1, 2], 1), (1, [1, 2])])
def test_result_type_invalid_scalar_pair(first, second):
    with pytest.raises((TypeError, RuntimeError)):
        _candidate(first, second)


@pytest.mark.result_type
def test_result_type_missing_argument():
    inp = _make_tensor(torch.float32, (4,))

    with pytest.raises((TypeError, RuntimeError)):
        _candidate(inp)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in ALL_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.result_type
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_result_type_shared_grid(dtype, shape, value_range):
    inp = _make_input(dtype, shape, value_range)
    other = _make_input(dtype, (), ["-1", "1"])
    ref = _reference(inp, other)

    res = _candidate(inp, other)

    _assert_result(res, ref)
