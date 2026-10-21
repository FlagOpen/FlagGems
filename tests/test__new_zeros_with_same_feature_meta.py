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

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _expected_device():
    device = torch.device(flag_gems.device)
    if device.index is None and device.type not in ("cpu", "meta"):
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return device


def _cpu_storage_view(inp):
    # Copy bytes before decoding: target FP8 casts and complex comparisons may
    # be absent. Retain the physical view instead of compacting its geometry.
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
    value = _cpu_storage_view(inp)
    if inp.dtype in _FP8_DTYPES:
        value = value.float()
    elif inp.dtype == torch.complex32:
        value = value.to(torch.complex64)
    if inp.is_conj():
        value = value.conj()
    if inp.is_neg():
        value = torch._neg_view(value)
    return value.resolve_conj().resolve_neg()


def _comparison_values(result, reference):
    # Check the original ABI before any CPU observation can erase it.
    assert isinstance(result, torch.Tensor) and isinstance(reference, torch.Tensor)
    assert result.device == _expected_device()
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape and result.layout == reference.layout
    assert reference.device == (torch.device("cpu") if utils.TO_CPU else result.device)
    if result.dtype in _FP8_DTYPES or (
        flag_gems.vendor_name in ("ascend", "kunlunxin") and result.dtype.is_complex
    ):
        return _numeric_values(result), _numeric_values(reference)
    return utils.to_cpu(result, reference), reference


def _assert_result_equal(result, reference):
    result, reference = _comparison_values(result, reference)
    torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)


def _to_reference(inp):
    if not utils.TO_CPU:
        return tu.to_reference(inp)
    reference = _cpu_storage_view(inp)
    if inp.is_conj():
        reference = reference.conj()
    if inp.is_neg():
        reference = torch._neg_view(reference)
    return reference.requires_grad_(inp.requires_grad)


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in _FP8_DTYPES:
        low, high = (tu.resolve_bound(bound, dtype) for bound in value_range)
        if low == high:
            value = _full(shape, low, dtype=dtype, device="cpu")
        else:
            value = torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        value = value.to(flag_gems.device)
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


def _full(shape, value, *, dtype, device):
    # Older CPU Torch can store FP8 but cannot fill it. Encode the sentinel
    # before transfer; the native zeros reference/candidate still run unchanged.
    if dtype in _FP8_DTYPES and torch.device(device).type == "cpu":
        return torch.full(shape, value, dtype=torch.float32, device="cpu").to(dtype)
    return torch.full(shape, value, dtype=dtype, device=device)


@pytest.fixture(autouse=True)
def _native_output_dtype_capability(request):
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    dtype = params.get("other_dtype", params.get("dtype"))
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        *_FP8_DTYPES,
        torch.complex32,
    ):
        pytest.skip(
            "Ascend/Kunlunxin native _new_zeros_with_same_feature_meta default/out does not support this output dtype (including empty outputs)"
        )


# Select every overload and contract case with the operator marker.
pytestmark = pytest.mark.underscore_new_zeros_with_same_feature_meta

# Allocate zeros with shape self.shape[:N] + other.shape and other's dtype/device.
# Cases are (self_shape, other_shape, N), where N is self_num_batch_dims.
_NEW_ZEROS_WITH_SAME_FEATURE_META_CASES = [
    pytest.param((2, 3, 4, 5), (7, 8, 9), 0, id="N0"),
    pytest.param((2, 3, 4, 5), (7, 8, 9), 1, id="N1"),
    pytest.param((2, 3, 4, 5), (7, 8, 9), 3, id="N3"),
    pytest.param((2, 3, 4, 5), (7, 8, 9), 4, id="N_self_rank"),
    pytest.param((2, 3, 4), (7, 8, 9), 3, id="self_3d_full"),
    pytest.param((2, 3), (4, 5, 6), 2, id="self_2d_other_3d"),
    pytest.param((3,), (4, 5), 1, id="self_1d"),
    pytest.param((3,), (4, 5), 0, id="self_1d_N0"),
    pytest.param((), (4, 5), 0, id="self_0d"),
    pytest.param((), (), 0, id="both_0d"),
    pytest.param((2, 3, 4), (5,), 3, id="other_1d"),
    pytest.param((2, 3), (), 2, id="other_0d"),
    pytest.param((0, 3), (4, 5), 1, id="self_zero_dim"),
    pytest.param((2, 3), (0, 5), 1, id="other_zero_dim"),
]

_NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
    + utils.COMPLEX_DTYPES
)

_NEW_ZEROS_WITH_SAME_FEATURE_META_MIXED_DTYPES = [
    pytest.param(torch.int16, torch.float16, id="self_int16_other_f16"),
    pytest.param(torch.float32, torch.bool, id="self_f32_other_bool"),
    pytest.param(torch.bool, torch.int32, id="self_bool_other_int32"),
    pytest.param(torch.int8, torch.float8_e4m3fn, id="self_int8_other_fp8"),
]

_VALUE_RANGE_CASES = [
    pytest.param((2, 3, 4), (5, 6), 1, id="self_3d_other_2d"),
    pytest.param((3,), (4, 5), 0, id="self_1d_N0"),
]

_MAIN_RANGE = ["-1", "1"]

_SHAPE_LEVEL_CASES = []
for shape in tu.selected_shapes():
    _SHAPE_LEVEL_CASES.append((shape, (4, 5), 0))
    if len(shape) >= 1:
        _SHAPE_LEVEL_CASES.append((shape, (4, 5), 1))
    if len(shape) >= 2:
        _SHAPE_LEVEL_CASES.append((shape, (2,), 2))
    _SHAPE_LEVEL_CASES.append(((2,), shape, 1))


def _special_tensor(shape, dtype, scenario):
    numel = math.prod(shape)
    values = _make_special_input(dtype, scenario)
    repeats = (numel + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:numel].reshape(shape)


def _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.device == other_t.device
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.requires_grad == ref_out.requires_grad
    _assert_result_equal(res_out, ref_out)
    # Default returns fresh storage. Explicit .out may alias an input, in
    # which case its writes and aliasing must agree with ATen.
    for inp, ref_inp in ((self_t, ref_self), (other_t, ref_other)):
        assert torch._C._is_alias_of(res_out, inp) == torch._C._is_alias_of(
            ref_out, ref_inp
        )
        _assert_result_equal(inp, ref_inp)


@pytest.mark.underscore_new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims",
    _NEW_ZEROS_WITH_SAME_FEATURE_META_CASES,
)
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta(
    self_shape, other_shape, self_num_batch_dims, dtype
):
    self_t = _make_input(dtype, self_shape, _MAIN_RANGE)
    other_t = _make_input(dtype, other_shape, _MAIN_RANGE)
    ref_self = _to_reference(self_t)
    ref_other = _to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark.new_zeros_with_same_feature_meta_out
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims",
    _NEW_ZEROS_WITH_SAME_FEATURE_META_CASES,
)
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta_out(
    self_shape, other_shape, self_num_batch_dims, dtype
):
    self_t = _make_input(dtype, self_shape, _MAIN_RANGE)
    other_t = _make_input(dtype, other_shape, _MAIN_RANGE)
    ref_self = _to_reference(self_t)
    ref_other = _to_reference(other_t)

    # Pre-sized out tensors with non-zero garbage values: the .out variant must
    # overwrite them in place with zeros and return the same object.
    expected_shape = self_shape[:self_num_batch_dims] + other_shape
    res_out = _full(expected_shape, 7, dtype=dtype, device=flag_gems.device)
    ref_out = _full(expected_shape, 7, dtype=dtype, device=ref_other.device)

    torch.ops.aten._new_zeros_with_same_feature_meta.out(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims, out=ref_out
    )
    res_ret = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims, out=res_out
    )

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is res_out
    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark.underscore_new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims", _SHAPE_LEVEL_CASES
)
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta_shapes(
    self_shape, other_shape, self_num_batch_dims, dtype
):
    self_t = _make_input(dtype, self_shape, _MAIN_RANGE)
    other_t = _make_input(dtype, other_shape, _MAIN_RANGE)
    ref_self = _to_reference(self_t)
    ref_other = _to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark.new_zeros_with_same_feature_meta_out
@pytest.mark.parametrize(
    "base_shape,stride,offset",
    [
        pytest.param((5, 4), (1, 4), 0, id="transposed"),
        pytest.param((4, 10), (10, 2), 0, id="strided"),
        pytest.param((5, 5), (5, 1), 5, id="offset"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES, quick=utils.FLOAT_DTYPES
    ),
)
def test__new_zeros_with_same_feature_meta_out_layouts(
    base_shape, stride, offset, dtype
):
    self_t = _make_input(dtype, (4, 10), _MAIN_RANGE)[:, ::2]
    other_t = _make_input(dtype, (8, 5), _MAIN_RANGE)[::2]
    ref_self, ref_other = _to_reference(self_t), _to_reference(other_t)
    res_base = _full(base_shape, 7, dtype=dtype, device=other_t.device)
    ref_base = _full(base_shape, 7, dtype=dtype, device=ref_other.device)
    res_out = res_base.as_strided((4, 5), stride, offset)
    ref_out = ref_base.as_strided((4, 5), stride, offset)

    torch.ops.aten._new_zeros_with_same_feature_meta.out(
        ref_self, ref_other, self_num_batch_dims=0, out=ref_out
    )
    res_ret = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=0, out=res_out
    )

    assert res_ret is res_out
    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)
    # The gaps and prefix outside the output view must remain untouched.
    _assert_result_equal(res_base, ref_base)


@pytest.mark.new_zeros_with_same_feature_meta_out
@pytest.mark.parametrize("out_shape", [(0,), (1,)])
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES, quick=utils.FLOAT_DTYPES
    ),
)
def test__new_zeros_with_same_feature_meta_out_resize(out_shape, dtype):
    self_t = _make_input(dtype, (2, 3), _MAIN_RANGE)
    other_t = _make_input(dtype, (4, 5), _MAIN_RANGE)
    ref_self, ref_other = _to_reference(self_t), _to_reference(other_t)
    res_out = _full(out_shape, 7, dtype=dtype, device=other_t.device)
    ref_out = _full(out_shape, 7, dtype=dtype, device=ref_other.device)

    torch.ops.aten._new_zeros_with_same_feature_meta.out(
        ref_self, ref_other, self_num_batch_dims=1, out=ref_out
    )
    res_ret = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=1, out=res_out
    )

    assert res_ret is res_out
    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark.new_zeros_with_same_feature_meta_out
@pytest.mark.parametrize("alias_self", [True, False])
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES, quick=utils.FLOAT_DTYPES
    ),
)
def test__new_zeros_with_same_feature_meta_out_alias(alias_self, dtype):
    self_t = _make_input(dtype, (4, 5), _MAIN_RANGE)
    other_t = _make_input(dtype, (4, 5), _MAIN_RANGE)
    ref_self, ref_other = _to_reference(self_t), _to_reference(other_t)
    res_out = self_t if alias_self else other_t
    ref_out = ref_self if alias_self else ref_other

    torch.ops.aten._new_zeros_with_same_feature_meta.out(
        ref_self, ref_other, self_num_batch_dims=0, out=ref_out
    )
    res_ret = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=0, out=res_out
    )

    assert res_ret is res_out
    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark.underscore_new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_grad,other_grad", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        [
            dtype
            for dtype in _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES
            if dtype.is_floating_point or dtype.is_complex
        ],
        quick=utils.FLOAT_DTYPES,
    ),
)
def test__new_zeros_with_same_feature_meta_no_autograd(self_grad, other_grad, dtype):
    self_t = _make_input(dtype, (2, 3), _MAIN_RANGE).requires_grad_(self_grad)
    other_t = _make_input(dtype, (4, 5), _MAIN_RANGE).requires_grad_(other_grad)
    ref_self, ref_other = _to_reference(self_t), _to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=1
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=1
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8 or value_range in (["-1", "1"], ["0", "max"])
]


@pytest.mark.underscore_new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims", _VALUE_RANGE_CASES
)
@pytest.mark.parametrize("dtype, value_range", _DTYPE_RANGE_CASES)
def test__new_zeros_with_same_feature_meta_value_ranges(
    self_shape, other_shape, self_num_batch_dims, value_range, dtype
):
    self_t = _make_input(dtype, self_shape, value_range)
    other_t = _make_input(dtype, other_shape, value_range)
    ref_self = _to_reference(self_t)
    ref_other = _to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark.underscore_new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_dtype, other_dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_MIXED_DTYPES
)
def test__new_zeros_with_same_feature_meta_other_dtype_wins(self_dtype, other_dtype):
    self_t = _make_input(self_dtype, (2, 3, 4), _MAIN_RANGE)
    other_t = _make_input(other_dtype, (7, 8), _MAIN_RANGE)
    ref_self = _to_reference(self_t)
    ref_other = _to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=1
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=1
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark.underscore_new_zeros_with_same_feature_meta
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta_same_tensor(dtype):
    self_t = _make_input(dtype, (2, 3, 4), _MAIN_RANGE)
    other_t = self_t
    ref_self = _to_reference(self_t)
    ref_other = _to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=1
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=1
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark.underscore_new_zeros_with_same_feature_meta
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)),
)
def test__new_zeros_with_same_feature_meta_nan_inf_values(shape, dtype, scenario):
    self_t = _special_tensor(shape, dtype, scenario)
    other_t = _special_tensor((4, 5), dtype, scenario)
    ref_self = _to_reference(self_t)
    ref_other = _to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=0
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=0
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark.underscore_new_zeros_with_same_feature_meta
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test__new_zeros_with_same_feature_meta_negative_batch_dims_raises(dtype):
    self_t = _make_input(dtype, (2, 3), _MAIN_RANGE)
    other_t = _make_input(dtype, (4, 5), _MAIN_RANGE)

    native_errors = (
        (RuntimeError, ValueError)
        if flag_gems.vendor_name == "kunlunxin"
        else RuntimeError
    )
    with pytest.raises(native_errors):
        torch.ops.aten._new_zeros_with_same_feature_meta(
            self_t, other_t, self_num_batch_dims=-1
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._new_zeros_with_same_feature_meta(
            self_t, other_t, self_num_batch_dims=-1
        )


@pytest.mark.new_zeros_with_same_feature_meta_out
def test__new_zeros_with_same_feature_meta_out_wrong_dtype_raises():
    self_t = _make_input(torch.float32, (2, 3), _MAIN_RANGE)
    other_t = _make_input(torch.float32, (4, 5), _MAIN_RANGE)
    out_t = _full((2, 4, 5), 1, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten._new_zeros_with_same_feature_meta.out(
            self_t, other_t, self_num_batch_dims=1, out=out_t
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._new_zeros_with_same_feature_meta(
            self_t, other_t, self_num_batch_dims=1, out=out_t
        )


@pytest.mark.underscore_new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_arg,other_arg",
    [
        pytest.param((1, 2), None, id="tuple_self"),
        pytest.param(1, None, id="int_self"),
        pytest.param(3.14, None, id="float_self"),
        pytest.param(None, 1, id="none_self"),
        pytest.param(None, None, id="none_both"),
    ],
)
def test__new_zeros_with_same_feature_meta_rejects_non_tensor(self_arg, other_arg):
    with pytest.raises(RuntimeError):
        torch.ops.aten._new_zeros_with_same_feature_meta(self_arg, other_arg)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._new_zeros_with_same_feature_meta(self_arg, other_arg)
