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
_NATIVE_OP_NAME = "_nested_tensor_size"


def _device(device=None):
    value = torch.device(flag_gems.device if device is None else device)
    if value.index is None and value.type != "cpu":
        value = torch.device(
            value.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return value


def _physical_bytes(inp):
    return (
        torch.empty(0, dtype=torch.uint8, device=inp.device)
        .set_(inp.untyped_storage(), 0, (inp.untyped_storage().nbytes(),), (1,))
        .detach()
        .cpu()
        .clone()
    )


def _physical_cpu(inp):
    raw = _physical_bytes(inp)
    return torch.empty(0, dtype=inp.dtype, device="cpu").set_(
        raw.untyped_storage(), inp.storage_offset(), inp.size(), inp.stride()
    )


def _geometry(inp):
    parts = inp.unbind()
    if parts:
        sizes = torch.tensor(
            [list(x.shape) for x in parts], dtype=torch.int64, device="cpu"
        )
        strides = torch.tensor(
            [list(x.stride()) for x in parts], dtype=torch.int64, device="cpu"
        )
    else:
        sizes = torch.empty((0, 0), dtype=torch.int64, device="cpu")
        strides = torch.empty((0, 0), dtype=torch.int64, device="cpu")
    offsets = torch.tensor(
        [x.storage_offset() for x in parts], dtype=torch.int64, device="cpu"
    )
    return parts, (sizes, strides, offsets)


def _cpu_shadow(inp):
    parts, metadata = _geometry(inp)
    raw = _physical_bytes(parts[0] if parts else inp.values())
    flat = torch.empty(0, dtype=inp.dtype, device="cpu").set_(
        raw.untyped_storage(), 0, (raw.numel() // inp.element_size(),), (1,)
    )
    return torch.ops.aten._nested_view_from_buffer(flat, *metadata)


def _make_dense_input(dtype, shape, value_range):
    if flag_gems.vendor_name == "ascend" and dtype in _FP8_DTYPES:
        pytest.skip(
            "Ascend cannot construct FP8 strided nested tensors (cat/fill); ordinary nested dtypes remain covered"
        )
    if flag_gems.vendor_name == "kunlunxin" and dtype in _FP8_DTYPES:
        low, high = (tu.resolve_bound(x, dtype) for x in value_range)
        if low == high:
            value = torch.full(shape, low, dtype=dtype, device="cpu")
        else:
            value = torch.testing.make_tensor(
                shape, dtype=dtype, device="cpu", low=low, high=high
            )
        return value.to(flag_gems.device)
    return tu.make_input(dtype, shape, value_range)


def _make_special_input(dtype, scenario):
    if flag_gems.vendor_name == "ascend" and dtype in _FP8_DTYPES:
        pytest.skip(
            "Ascend cannot construct FP8 strided nested tensors (cat/fill); ordinary nested dtypes remain covered"
        )
    if flag_gems.vendor_name == "kunlunxin" and dtype in _FP8_DTYPES:
        payload = {
            "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
            "inf": [float("inf"), -float("inf"), 0.0, -0.0, 1.0],
            "mixed": [float("nan"), float("inf"), -float("inf"), 0.0, -0.0],
        }[scenario]
        return (
            torch.tensor(payload, dtype=torch.float32, device="cpu")
            .to(dtype)
            .to(flag_gems.device)
        )
    return tu.make_special_input(dtype, scenario)


def _repeat_special(special, repeats, numel, length):
    if special.dtype in _FP8_DTYPES:
        # Replicate payload bytes without backend FP8 arithmetic/casting.
        raw = _physical_cpu(special).view(torch.uint8).repeat(repeats)[:numel]
        return raw.view(special.dtype).reshape(length, 4).to(special.device)
    return special.repeat(repeats)[:numel].reshape(length, 4)


def _nested(components, *, dtype, device=None):
    target = _device(device)
    if flag_gems.vendor_name == "ascend" and dtype in _FP8_DTYPES:
        pytest.skip(
            "Ascend cannot construct FP8 strided nested tensors (cat/fill); ordinary nested dtypes remain covered"
        )
    if (
        components
        and dtype in _FP8_DTYPES
        and (
            flag_gems.vendor_name in ("hygon", "kunlunxin", "metax")
            or target.type == "cpu"
        )
    ):
        # The operator queries metadata, while legacy GPU cat lacks FP8.
        # Pack identical bytes and attach the original contiguous component layouts.
        host = [_physical_cpu(x) for x in components]
        packed = torch.cat(
            [x.contiguous().view(torch.uint8).reshape(-1) for x in host]
        ).view(dtype)
        sizes = torch.tensor(
            [list(x.shape) for x in host], dtype=torch.int64, device="cpu"
        )
        strides = torch.tensor(
            [list(x.stride()) for x in host], dtype=torch.int64, device="cpu"
        )
        offsets = []
        total = 0
        for part in host:
            offsets.append(total)
            total += part.numel()
        nested = torch.ops.aten._nested_view_from_buffer(
            packed.to(target),
            sizes,
            strides,
            torch.tensor(offsets, dtype=torch.int64, device="cpu"),
        )
    else:
        nested = torch.nested.nested_tensor(components, dtype=dtype, device=target)
    assert nested.is_nested and nested.dtype == dtype and nested.device == target
    return nested


def _to_reference(inp):
    if (
        not isinstance(inp, torch.Tensor)
        or not inp.is_nested
        or inp.layout != torch.strided
    ):
        return tu.to_reference(inp)
    if utils.TO_CPU:
        return _cpu_shadow(inp)
    if flag_gems.vendor_name == "ascend":
        # NestedNPU lacks detach/clone/to_copy; reconstruct an independent input
        # with the same component payloads, not a shared reference alias.
        parts = [_physical_cpu(x).to(inp.device) for x in inp.unbind()]
        return _nested(parts, dtype=inp.dtype, device=inp.device)
    if inp.dtype in _FP8_DTYPES and flag_gems.vendor_name in (
        "hygon",
        "kunlunxin",
        "metax",
    ):
        return _nested(list(inp.unbind()), dtype=inp.dtype, device=inp.device)
    return tu.to_reference(inp)


def cpu_oracle(inp):
    """Host metadata oracle; never transfers a nested tensor through .cpu()."""
    native = getattr(torch.ops.aten, _NATIVE_OP_NAME)
    if not isinstance(inp, torch.Tensor):
        return native(inp)
    if not inp.is_nested or inp.layout != torch.strided:
        raise NotImplementedError("strided-layout nested tensor required")
    if inp.device.type == "cpu":
        return native(inp)
    _, metadata = _geometry(inp)
    index = {
        "_nested_tensor_size": 0,
        "_nested_tensor_strides": 1,
        "_nested_tensor_storage_offsets": 2,
    }[_NATIVE_OP_NAME]
    return metadata[index]


def _reference(inp):
    if flag_gems.vendor_name == "ascend" and _NATIVE_OP_NAME != "_nested_tensor_size":
        return cpu_oracle(inp)
    return getattr(torch.ops.aten, _NATIVE_OP_NAME)(inp)


def _dense_identity(inp):
    storage = inp.untyped_storage()
    return (
        storage._cdata,
        storage.data_ptr(),
        storage.nbytes(),
        inp.device,
        inp.dtype,
        inp.layout,
        tuple(inp.shape),
        inp.stride(),
        inp.storage_offset(),
        inp.is_neg(),
        inp.is_conj(),
        inp.requires_grad,
    )


def _candidate(inp):
    if not isinstance(inp, torch.Tensor):
        return getattr(flag_gems, _NATIVE_OP_NAME)(inp)
    abi = (inp.device, inp.dtype, inp.layout, inp.requires_grad)
    if not inp.is_nested:
        parts, metadata = [inp], ()
    elif inp.layout == torch.strided:
        parts, metadata = _geometry(inp)
        if not parts:
            parts = [inp.values()]
    else:
        # Even an unsupported jagged input must remain read-only on error.
        parts = [inp.values(), inp.offsets()]
        if inp.lengths() is not None:
            parts.append(inp.lengths())
        metadata = ()
    identities = [_dense_identity(x) for x in parts]
    payloads = [_physical_bytes(x) for x in parts]
    try:
        result = getattr(flag_gems, _NATIVE_OP_NAME)(inp)
    finally:
        assert (inp.device, inp.dtype, inp.layout, inp.requires_grad) == abi
        if not inp.is_nested:
            after, after_metadata = [inp], ()
        elif inp.layout == torch.strided:
            after, after_metadata = _geometry(inp)
            if not after:
                after = [inp.values()]
        else:
            after = [inp.values(), inp.offsets()]
            if inp.lengths() is not None:
                after.append(inp.lengths())
            after_metadata = ()
        assert len(after) == len(parts)
        assert [_dense_identity(x) for x in after] == identities
        for value, payload in zip(after, payloads):
            torch.testing.assert_close(_physical_bytes(value), payload, rtol=0, atol=0)
        for before_value, after_value in zip(metadata, after_metadata):
            torch.testing.assert_close(after_value, before_value, rtol=0, atol=0)
    return result


# Read the CPU int64 metadata of a strided-layout nested tensor.
# Empty batches and .out are excluded because the current reference cannot run them.
_NUM_TENSORS = [1, 8, 64]

_VALUE_RANGE = ["-1", "1"]

_TRAILING_EXTENT = 4

_MAX_TRAILING_EXTENT = 4

_COMPONENT_DTYPES = (
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
    + utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool]
)

_NEGATIVE_EXC = (
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    IndexError,
)

_COMPONENT_RANKS = sorted(
    {len(shape) for shape in tu.selected_shapes() if len(shape) >= 1}
)

# (shape, trailing dimensions); cap trailing extents while retaining component rank.
_SHAPE_LEVEL_CASES = [
    (shape, tuple(min(int(dim), _MAX_TRAILING_EXTENT) for dim in shape[1:]))
    for shape in tu.selected_shapes()
    if len(shape) >= 1
]

_VALUE_RANGE_LAYOUTS = tu.selected_cases([(8, 2), (32, 3), (64, 4)], quick=[(8, 2)])


def _make_nested(
    num_tensors,
    num_dims,
    dtype,
    value_range=_VALUE_RANGE,
    seed=0,
    trailing=_TRAILING_EXTENT,
):
    # Vary the leading extent per component; keep trailing dimensions fixed.
    gen = torch.Generator("cpu").manual_seed(seed)
    lengths = torch.randint(1, 9, (num_tensors,), generator=gen, device="cpu").tolist()
    components = [
        _make_dense_input(dtype, (length,) + (trailing,) * (num_dims - 1), value_range)
        for length in lengths
    ]
    inp = _nested(components, dtype=dtype)
    return inp


def _assert_sizes(res_out, ref_out):
    assert isinstance(res_out, torch.Tensor) and res_out.dtype == torch.int64
    assert ref_out.device == torch.device("cpu") and ref_out.dtype == torch.int64
    assert res_out.shape == ref_out.shape and res_out.layout == ref_out.layout
    assert res_out.device == torch.device("cpu")
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.underscore_nested_tensor_size
@pytest.mark.parametrize("num_tensors", _NUM_TENSORS)
@pytest.mark.parametrize("num_dims", _COMPONENT_RANKS)
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_size(num_tensors, num_dims, dtype):
    inp = _make_nested(num_tensors, num_dims, dtype)
    ref_inp = _to_reference(inp)

    ref_out = _reference(ref_inp)
    res_out = _candidate(inp)

    _assert_sizes(res_out, ref_out)


@pytest.mark.underscore_nested_tensor_size
@pytest.mark.parametrize("case", _SHAPE_LEVEL_CASES)
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_size_shape_levels(case, dtype):
    _, trailing = case
    num_tensors = 3
    gen = torch.Generator("cpu").manual_seed(0)
    lengths = torch.randint(1, 5, (num_tensors,), generator=gen, device="cpu").tolist()
    components = [
        _make_dense_input(dtype, (length,) + trailing, _VALUE_RANGE)
        for length in lengths
    ]
    inp = _nested(components, dtype=dtype)
    ref_inp = _to_reference(inp)

    ref_out = _reference(ref_inp)
    res_out = _candidate(inp)

    _assert_sizes(res_out, ref_out)


@pytest.mark.underscore_nested_tensor_size
@pytest.mark.parametrize("case", _VALUE_RANGE_LAYOUTS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_size_value_ranges(case, value_range, dtype):
    num_tensors, num_dims = case
    inp = _make_nested(num_tensors, num_dims, dtype, value_range=value_range)
    ref_inp = _to_reference(inp)

    ref_out = _reference(ref_inp)
    res_out = _candidate(inp)

    _assert_sizes(res_out, ref_out)


@pytest.mark.underscore_nested_tensor_size
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_size_uniform(dtype):
    num_tensors = 6
    components = [
        _make_dense_input(dtype, (4, 4, 4), _VALUE_RANGE) for _ in range(num_tensors)
    ]
    inp = _nested(components, dtype=dtype)
    assert inp.is_nested
    ref_inp = _to_reference(inp)

    ref_out = _reference(ref_inp)
    res_out = _candidate(inp)

    _assert_sizes(res_out, ref_out)


@pytest.mark.underscore_nested_tensor_size
@pytest.mark.parametrize("dtype", _COMPONENT_DTYPES)
def test__nested_tensor_size_with_empty_components(dtype):
    num_tensors = 4
    gen = torch.Generator("cpu").manual_seed(1)
    lengths = [
        int(torch.randint(0, 4, (1,), generator=gen, device="cpu").item())
        for _ in range(num_tensors)
    ]
    components = [
        _make_dense_input(dtype, (length, 4), _VALUE_RANGE) for length in lengths
    ]
    inp = _nested(components, dtype=dtype)
    ref_inp = _to_reference(inp)

    ref_out = _reference(ref_inp)
    res_out = _candidate(inp)

    _assert_sizes(res_out, ref_out)


@pytest.mark.underscore_nested_tensor_size
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_COMPONENT_DTYPES))
)
def test__nested_tensor_size_nan_inf_values(dtype, scenario):
    num_tensors = 4
    gen = torch.Generator("cpu").manual_seed(3)
    lengths = torch.randint(1, 5, (num_tensors,), generator=gen, device="cpu").tolist()
    special = _make_special_input(dtype, scenario)
    components = []
    for length in lengths:
        numel = length * 4
        repeats = (numel + special.numel() - 1) // special.numel()
        values = _repeat_special(special, repeats, numel, length)
        components.append(values)
    inp = _nested(components, dtype=dtype)
    ref_inp = _to_reference(inp)

    ref_out = _reference(ref_inp)
    res_out = _candidate(inp)

    _assert_sizes(res_out, ref_out)


@pytest.mark.underscore_nested_tensor_size
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__nested_tensor_size_dense_raises(dtype):
    inp = _make_dense_input(dtype, (4, 4), _VALUE_RANGE)
    with pytest.raises(NotImplementedError):
        _reference(_to_reference(inp))
    with pytest.raises(_NEGATIVE_EXC):
        _candidate(inp)


@pytest.mark.underscore_nested_tensor_size
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__nested_tensor_size_jagged_raises(dtype):
    ref_device = torch.device("cpu") if utils.TO_CPU else flag_gems.device
    offsets = torch.tensor([0, 2, 5, 9], dtype=torch.int64)
    values = _make_dense_input(dtype, (9, 2), _VALUE_RANGE)
    inp = torch.nested.nested_tensor_from_jagged(values, offsets.to(flag_gems.device))
    ref_inp = torch.nested.nested_tensor_from_jagged(
        values.to(ref_device), offsets.to(ref_device)
    )
    assert inp.layout == torch.jagged
    with pytest.raises(NotImplementedError):
        _reference(ref_inp)
    with pytest.raises(_NEGATIVE_EXC):
        _candidate(inp)


@pytest.mark.underscore_nested_tensor_size
def test__nested_tensor_size_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        _reference(3.14)
    with pytest.raises(_NEGATIVE_EXC):
        _candidate(3.14)
