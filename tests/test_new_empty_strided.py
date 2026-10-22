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


def _to_reference(value):
    if utils.TO_CPU:
        return _cpu_copy(value).requires_grad_(value.requires_grad)
    return tu.to_reference(value)


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
    value = tu.make_input(dtype, shape, value_range)
    assert value.dtype == dtype and value.device == device
    assert tuple(value.shape) == tuple(shape)
    return value


def _make_special_input(dtype, scenario):
    device = _device(flag_gems.device)
    value = tu.make_special_input(dtype, scenario)
    assert value.dtype == dtype and value.device == device
    assert tuple(value.shape) == (5,)
    return value


def _empty(*args, dtype, device):
    device = _device(device)
    value = torch.empty(*args, dtype=dtype, device=device)
    assert value.dtype == dtype and value.device == device
    return value


def _fill(value, scalar):
    if value.dtype in _FP8_DTYPES or (
        flag_gems.vendor_name == "kunlunxin" and value.dtype.is_complex
    ):
        encoded = torch.full(value.shape, scalar, dtype=torch.float32, device="cpu").to(
            value.dtype
        )
        value.copy_(encoded.to(value.device))
    else:
        value.fill_(scalar)


def _candidate(
    inp, size, stride, *, required_device, expected_dtype, observe_input=True, **kwargs
):
    assert inp.device == required_device and inp.dtype == expected_dtype
    original = _signature(inp)
    storage = inp.untyped_storage()
    base = inp._base
    base_storage = None if base is None else base.untyped_storage()
    base_signature = None if base is None else _signature(base)
    before = _input_bytes(inp) if observe_input else None
    out = kwargs.get("out")
    output_device = (
        out.device
        if out is not None
        else _device(kwargs.get("device") or required_device)
    )
    output_dtype = (
        out.dtype if out is not None else (kwargs.get("dtype") or expected_dtype)
    )
    result = flag_gems.new_empty_strided(inp, size, stride, **kwargs)
    assert isinstance(result, torch.Tensor)
    assert result.device == output_device and result.dtype == output_dtype
    assert (
        _signature(inp) == original and inp.untyped_storage()._cdata == storage._cdata
    )
    assert inp._base is base
    if base is not None:
        assert _signature(base) == base_signature
        assert base.untyped_storage()._cdata == base_storage._cdata
    if observe_input:
        assert torch.equal(_input_bytes(inp), before)
    if out is not None:
        assert result is out
    else:
        assert result is not inp and not result._is_view()
        assert result.untyped_storage()._cdata != storage._cdata
        assert result.layout == torch.strided and result.storage_offset() == 0
        assert tuple(result.shape) == tuple(size) and tuple(result.stride()) == tuple(
            stride
        )
        if result.numel():
            assert result.data_ptr() != inp.data_ptr()
    return result


def _assert_result_equal(result, reference, *, required_device):
    assert result.device == required_device
    assert reference.device == (
        torch.device("cpu") if utils.TO_CPU else required_device
    )
    assert result.dtype == reference.dtype and result.layout == reference.layout
    assert result.shape == reference.shape

    def observe(value):
        copy = _cpu_copy(value)
        physical = (
            torch.empty(0, dtype=torch.uint8, device="cpu")
            .view(copy.dtype)
            .set_(
                copy.untyped_storage(), copy.storage_offset(), copy.shape, copy.stride()
            )
        )
        if physical.dtype in _FP8_DTYPES:
            physical = physical.float()
        if physical.dtype == torch.complex32:
            physical = physical.to(torch.complex64)
        if value.is_conj():
            physical = physical.conj()
        if value.is_neg():
            physical = -physical
        return physical

    torch.testing.assert_close(
        observe(result), observe(reference), rtol=0, atol=0, equal_nan=True
    )


# aten::new_empty_strided allocates uninitialized storage for an explicit size and
# stride; `self` contributes dtype/device only and is never aliased. The normal path
# computes no element, so the result payload is undefined and is read only under
# deterministic fill, where the allocator defines it. `size` and `stride` are sequence
# arguments, so there is no operand broadcast, no scalar-operand form and no backward
# workload: the result is a fresh allocation, not a function of `self`.

_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    # Static capability flags, read at import time: no tensor is allocated, no operator
    # is called, and nothing is probed or skipped at run time.
    flag = _DTYPE_CAPABILITY.get(dtype)
    return (
        True if flag is None else bool(getattr(flag_gems.runtime.device, flag, False))
    )


# complex64 rides the 32-bit float path and needs no capability flag of its own.
SUPPORTED_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES
    + [torch.float64, torch.bool, torch.complex64, torch.complex128]
    if _dtype_supported(dtype)
]

# Zero-extent requests are valid and keep their explicit stride metadata.
_ZERO_SHAPES = [(0,), (2, 0, 3)]
_LAYOUTS = ["contiguous", "spaced", "zero_stride"]


def _contiguous_stride(size):
    running = 1
    strides = []
    for dim in reversed(size):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))


def _stride_for(size, layout):
    dense = _contiguous_stride(size)
    if layout == "contiguous":
        return dense
    if layout == "spaced":
        # Doubled contiguous strides: gaps between elements are valid metadata.
        return tuple(step * 2 for step in dense)
    if layout == "zero_stride":
        # A zero outer stride is valid for every non-scalar rank, rank 1 and
        # zero-extent requests included; a 0-dim tensor's stride is the empty tuple.
        return () if not size else (0,) + dense[1:]
    raise ValueError("unsupported layout " + repr(layout))


def _layouts_for(size):
    if not size:
        return ["contiguous"]
    return list(_LAYOUTS)


def _assert_allocation(res_out, ref_out, inp, size, stride, dtype):
    # The payload is undefined, so only the metadata the native operator defines is
    # compared; the native storage byte size already is the element span.
    assert tuple(res_out.shape) == tuple(size)
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.stride() == ref_out.stride() == tuple(stride)
    assert res_out.dtype == dtype
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == inp.device
    assert res_out.layout == torch.strided
    assert res_out.storage_offset() == 0
    assert res_out._is_view() is False
    assert res_out.untyped_storage().nbytes() == ref_out.untyped_storage().nbytes()
    if res_out.numel():
        assert res_out.data_ptr() != inp.data_ptr()


# One row per (requested size, requested stride layout).
_ALL_ROWS = [
    (tuple(shape), layout)
    for shape in list(tu.selected_shapes()) + _ZERO_SHAPES
    for layout in _layouts_for(tuple(shape))
]
# Every cheap boundary stays in quick: all small shapes and the zero-extent requests
# with each layout, across the full dtype set.
_QUICK_SHAPES = [(), (1,), (256,), (2, 19, 7), (0,), (2, 0, 3)]
_QUICK_ROWS = [
    (tuple(shape), layout)
    for shape in _QUICK_SHAPES
    for layout in _layouts_for(tuple(shape))
]
_GRID_ROWS = tu.selected_cases(_ALL_ROWS, quick=_QUICK_ROWS)


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in SUPPORTED_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8 or value_range in (["-1", "1"], ["0", "max"])
]


@pytest.mark.new_empty_strided
@pytest.mark.parametrize(
    "shape,layout,dtype,value_range",
    [
        pytest.param(
            shape,
            layout,
            dtype,
            value_range,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and str(dtype).startswith("torch.float8_")
                and 0 not in shape,
                reason="Ascend cannot construct nonempty FP8 inputs with device casts or fills",
            ),
            # Keep the original IDs while marking dtype and source shape together.
            id=f"dtype{case_index}-value_range{case_index}-shape{shape_index}-{layout}",
        )
        for case_index, (dtype, value_range) in enumerate(_DTYPE_RANGE_CASES)
        for shape_index, (shape, layout) in enumerate(_GRID_ROWS)
    ],
)
def test_new_empty_strided_allocation(shape, layout, value_range, dtype):
    required_device = _device(flag_gems.device)
    size = tuple(shape)
    stride = _stride_for(size, layout)
    inp = _make_input(dtype, size, value_range)
    inp_before = _to_reference(inp)

    ref_out = torch.ops.aten.new_empty_strided(inp_before, list(size), list(stride))
    res_out = _candidate(
        inp,
        list(size),
        list(stride),
        required_device=required_device,
        expected_dtype=dtype,
    )

    _assert_allocation(res_out, ref_out, inp, size, stride, dtype)
    _assert_result_equal(inp, inp_before, required_device=required_device)


# (requested size, stride, buffer kind, storage retained). The native out
# overload overwrites even retained buffers with undefined data in ordinary
# mode. Only deterministic fill defines the result payload.
_OUT_ROWS = [
    ((2, 3), (3, 1), "matching", True),
    ((0, 3), (3, 1), "matching", True),
    ((), (), "matching", True),
    ((3, 1), (1, 1), "matching", True),
    ((4, 3), (1, 4), "transposed", True),
    ((2, 3), (6, 2), "stride_ignored", True),
    ((2, 3), (3, 1), "int8_buffer", True),
    ((2, 3), (3, 1), "flat_buffer", True),
    ((2, 3), (6, 2), "zero_size_buffer", False),
    ((2, 3), (3, 1), "offset_window", True),
]


def _make_out_buffer(kind, size, dtype, device):
    base = None
    if kind == "transposed":
        buf = _empty(size[1], size[0], dtype=dtype, device=device).t()
    elif kind == "int8_buffer":
        # The out buffer's dtype is authoritative for the result.
        buf = _empty(size, dtype=torch.int8, device=device)
    elif kind == "flat_buffer":
        buf = _empty(math.prod(size), dtype=dtype, device=device)
    elif kind == "zero_size_buffer":
        buf = _empty(0, dtype=dtype, device=device)
    elif kind == "offset_window":
        # A window inside a larger allocation; the guard bytes around it are outside
        # the requested extent and are compared through the parent storage.
        base = _empty(math.prod(size) + 2, dtype=dtype, device=device)
        buf = base[2:]
    else:
        buf = _empty(size, dtype=dtype, device=device)
    if base is not None:
        _fill(base, False if base.dtype == torch.bool else 7)
    _fill(buf, False if buf.dtype == torch.bool else 7)
    return buf, base


@pytest.mark.new_empty_strided
@pytest.mark.parametrize(
    "deterministic",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend",
                reason="Ascend native allocation does not implement deterministic uninitialized fill",
            ),
        ),
    ],
)
@pytest.mark.parametrize("size,stride,kind,keeps_storage", _OUT_ROWS)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and str(dtype).startswith("torch.float8_"),
                reason="Ascend cannot construct FP8 inputs with device casts or fills",
            ),
        )
        for dtype in SUPPORTED_DTYPES
    ],
)
def test_new_empty_strided_out_reuses_buffer(
    size, stride, kind, keeps_storage, dtype, deterministic
):
    required_device = _device(flag_gems.device)
    output_dtype = torch.int8 if kind == "int8_buffer" else dtype
    if (
        deterministic
        and flag_gems.vendor_name == "hygon"
        and output_dtype in _FP8_DTYPES
    ):
        pytest.skip(
            "Hygon Torch native deterministic fill is unavailable for FP8 output storage"
        )
    if deterministic and flag_gems.vendor_name == "kunlunxin" and math.prod(size) > 0:
        pytest.skip(
            "Kunlunxin native new_empty_strided.out does not implement this deterministic fill"
        )
    inp = _make_input(dtype, (3, 4), ["-1", "1"])
    inp_before = _to_reference(inp)
    buf, base = _make_out_buffer(kind, size, dtype, inp.device)
    ref_buf, ref_base = _make_out_buffer(kind, size, dtype, inp_before.device)
    original_out_storage = buf.untyped_storage()
    storage_ptr = original_out_storage.data_ptr()
    storage_nbytes = buf.untyped_storage().nbytes()

    # Distinct sentinels detect writes; deterministic mode defines newly
    # allocated bytes, while ordinary mode checks geometry and guard bytes.
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        ref_out = torch.ops.aten.new_empty_strided.out(
            inp_before, list(size), list(stride), out=ref_buf
        )
        res_out = _candidate(
            inp,
            list(size),
            list(stride),
            out=buf,
            required_device=required_device,
            expected_dtype=dtype,
        )
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )

    # The out overload hands back the buffer object itself.
    assert res_out is buf
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.stride() == ref_out.stride()
    assert res_out.dtype == ref_out.dtype
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.device == inp.device
    if keeps_storage:
        assert res_out.untyped_storage()._cdata == original_out_storage._cdata
        # A buffer that already satisfies the request retains its storage; a zero-size
        # buffer is reallocated by the resize instead.
        assert res_out.untyped_storage().data_ptr() == storage_ptr
        assert res_out.untyped_storage().nbytes() == storage_nbytes
    if deterministic:
        _assert_result_equal(res_out, ref_out, required_device=required_device)
    if base is not None:
        if deterministic:
            _assert_result_equal(base, ref_base, required_device=required_device)
        else:
            _assert_result_equal(
                base[:2], ref_base[:2], required_device=required_device
            )
    _assert_result_equal(inp, inp_before, required_device=required_device)


# Optional-keyword coverage. pin_memory=True is rejected for a non-CPU source and is a
# negative row below; its valid CPU form is a separate test.
_KWARG_ROWS = [
    ("strided_layout", {"layout": torch.strided}),
    ("explicit_device", {"device": flag_gems.device}),
    ("pin_memory_false", {"pin_memory": False}),
]


@pytest.mark.new_empty_strided
@pytest.mark.parametrize(
    "kwargs",
    [row[1] for row in _KWARG_ROWS],
    ids=[row[0] for row in _KWARG_ROWS],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and str(dtype).startswith("torch.float8_"),
                reason="Ascend cannot construct FP8 inputs with device casts or fills",
            ),
        )
        for dtype in SUPPORTED_DTYPES
    ],
)
def test_new_empty_strided_optional_kwargs(kwargs, dtype):
    required_device = _device(flag_gems.device)
    size, stride = (2, 19, 7), (133, 7, 1)
    inp = _make_input(dtype, size, ["-1", "1"])
    inp_before = _to_reference(inp)

    ref_out = torch.ops.aten.new_empty_strided(
        inp_before, list(size), list(stride), **kwargs
    )
    res_out = _candidate(
        inp,
        list(size),
        list(stride),
        **kwargs,
        required_device=required_device,
        expected_dtype=dtype,
    )

    _assert_allocation(res_out, ref_out, inp, size, stride, dtype)
    _assert_result_equal(inp, inp_before, required_device=required_device)


_DTYPE_OVERRIDE_CASES = [
    dtype
    for dtype in [torch.int32, torch.uint8, torch.float64, torch.bool]
    if _dtype_supported(dtype)
]


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("override", _DTYPE_OVERRIDE_CASES, ids=lambda d: str(d))
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and str(dtype).startswith("torch.float8_"),
                reason="Ascend cannot construct FP8 inputs with device casts or fills",
            ),
        )
        for dtype in SUPPORTED_DTYPES
    ],
)
def test_new_empty_strided_dtype_override(override, dtype):
    required_device = _device(flag_gems.device)
    size, stride = (2, 19, 7), (133, 7, 1)
    inp = _make_input(dtype, (2, 2), ["-1", "1"])
    inp_before = _to_reference(inp)

    ref_out = torch.ops.aten.new_empty_strided(
        inp_before, list(size), list(stride), dtype=override
    )
    res_out = _candidate(
        inp,
        list(size),
        list(stride),
        dtype=override,
        required_device=required_device,
        expected_dtype=dtype,
    )

    _assert_allocation(res_out, ref_out, inp, size, stride, override)
    _assert_result_equal(inp, inp_before, required_device=required_device)


# Pinning is defined for a dense CPU source: both the oracle and the candidate receive
# the same explicit CPU tensor and the same requested CPU device.
@pytest.mark.new_empty_strided
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_new_empty_strided_pin_memory_on_cpu_source(dtype):
    required_device = torch.device("cpu")
    size, stride = (2, 19, 7), (133, 7, 1)
    inp = torch.empty(size, dtype=dtype, device="cpu")
    ref_inp = inp.clone()

    ref_out = torch.ops.aten.new_empty_strided(
        ref_inp, list(size), list(stride), pin_memory=True, device=torch.device("cpu")
    )
    res_out = _candidate(
        inp,
        list(size),
        list(stride),
        pin_memory=True,
        device=torch.device("cpu"),
        required_device=required_device,
        expected_dtype=dtype,
        observe_input=False,
    )

    assert res_out.is_pinned()
    assert tuple(res_out.shape) == tuple(ref_out.shape) == size
    assert res_out.stride() == ref_out.stride() == tuple(stride)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device.type == "cpu"
    assert res_out.untyped_storage().nbytes() == ref_out.untyped_storage().nbytes()


# `self` is metadata-only, so a transposed, offset-window, expanded or
# gapped source cannot change the allocation.
_SOURCE_LAYOUT_ROWS = [
    ("transposed", (4, 6)),
    ("offset_window", (5, 6)),
    ("expanded", (1, 6)),
    ("column_step", (4, 12)),
]


def _apply_source_layout(base, layout):
    if layout == "transposed":
        return base.t()
    if layout == "offset_window":
        return base[2:5]
    if layout == "expanded":
        return base.expand(3, base.shape[-1])
    if layout == "column_step":
        return base[:, ::2]
    raise ValueError("unsupported source layout " + repr(layout))


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("layout,storage_shape", _SOURCE_LAYOUT_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_new_empty_strided_source_layout_independent(layout, storage_shape, dtype):
    required_device = _device(flag_gems.device)
    size, stride = (3, 4), (4, 1)
    base = _make_input(dtype, storage_shape, ["-1", "1"])
    ref_base = _to_reference(base)
    inp = _apply_source_layout(base, layout)
    ref_inp = _apply_source_layout(ref_base, layout)
    base_before = _to_reference(base)

    ref_out = torch.ops.aten.new_empty_strided(ref_inp, list(size), list(stride))
    res_out = _candidate(
        inp,
        list(size),
        list(stride),
        required_device=required_device,
        expected_dtype=dtype,
    )

    _assert_allocation(res_out, ref_out, inp, size, stride, dtype)
    _assert_result_equal(base, base_before, required_device=required_device)


# Default-only: the source carries special values, but it contributes dtype/device
# only, so its payload must neither propagate to the undefined result nor be written.
_SPECIAL_SOURCE_CASES = tu.selected_cases(
    tu.special_value_cases(
        [dtype for dtype in SUPPORTED_DTYPES if dtype.is_floating_point]
    ),
    quick=[],
)


@pytest.mark.new_empty_strided
@pytest.mark.parametrize(
    "dtype,scenario",
    [
        pytest.param(
            dtype,
            scenario,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and str(dtype).startswith("torch.float8_"),
                reason="Ascend cannot construct FP8 inputs with device casts or fills",
            ),
        )
        for dtype, scenario in _SPECIAL_SOURCE_CASES
    ],
)
def test_new_empty_strided_special_source_values(dtype, scenario):
    required_device = _device(flag_gems.device)
    size, stride = (2, 3), (3, 1)
    inp = _make_special_input(dtype, scenario)
    inp_before = _to_reference(inp)

    ref_out = torch.ops.aten.new_empty_strided(inp_before, list(size), list(stride))
    res_out = _candidate(
        inp,
        list(size),
        list(stride),
        required_device=required_device,
        expected_dtype=dtype,
    )

    _assert_allocation(res_out, ref_out, inp, size, stride, dtype)
    _assert_result_equal(inp, inp_before, required_device=required_device)


# A fresh allocation owns its storage: writing the result (a write only, the undefined
# payload is never read) must not reach the source.
_ISOLATION_ROWS = [((2, 3), "contiguous"), ((2, 3), "spaced"), ((3, 4), "zero_stride")]


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("shape,layout", _ISOLATION_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_new_empty_strided_result_storage_isolated(shape, layout, dtype):
    required_device = _device(flag_gems.device)
    size = tuple(shape)
    stride = _stride_for(size, layout)
    inp = _make_input(dtype, size, ["-1", "1"])
    inp_before = _to_reference(inp)

    res_out = _candidate(
        inp,
        list(size),
        list(stride),
        required_device=required_device,
        expected_dtype=dtype,
    )
    res_out.fill_(1)

    _assert_result_equal(inp, inp_before, required_device=required_device)


@pytest.mark.new_empty_strided
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "ascend"
                and str(dtype).startswith("torch.float8_"),
                reason="Ascend cannot construct FP8 inputs with device casts or fills",
            ),
        )
        for dtype in SUPPORTED_DTYPES
        if dtype.is_floating_point or dtype.is_complex
    ],
)
def test_new_empty_strided_result_has_no_autograd(dtype):
    required_device = _device(flag_gems.device)
    inp = _make_input(dtype, (3, 4), ["-1", "1"]).requires_grad_(True)
    inp_before = _to_reference(inp)

    res_out = _candidate(
        inp, [2, 3], [3, 1], required_device=required_device, expected_dtype=dtype
    )

    # An allocation is not a function of `self`, so it stays out of the graph.
    assert res_out.requires_grad is False
    assert res_out.grad_fn is None
    _assert_result_equal(inp, inp_before, required_device=required_device)


@pytest.mark.new_empty_strided
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend native allocation does not implement deterministic uninitialized fill",
)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_new_empty_strided_deterministic_fill(dtype):
    if flag_gems.vendor_name == "kunlunxin":
        pytest.skip(
            "Kunlunxin native new_empty_strided does not implement deterministic uninitialized fill"
        )
    if flag_gems.vendor_name == "hygon" and dtype in _FP8_DTYPES:
        pytest.skip(
            "Hygon Torch native deterministic fill is unavailable for FP8 output storage"
        )
    # Deterministic algorithms with fill_uninitialized_memory are the one path that
    # defines the payload of a fresh allocation, so the two results are compared
    # exactly (NaN for floats, the dtype maximum for integers, True for bool).
    required_device = _device(flag_gems.device)
    size, stride = (2, 3), (3, 1)
    inp = _make_input(dtype, (2, 2), ["-1", "1"])
    ref_inp = _to_reference(inp)

    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        ref_out = torch.ops.aten.new_empty_strided(ref_inp, list(size), list(stride))
        res_out = _candidate(
            inp,
            list(size),
            list(stride),
            required_device=required_device,
            expected_dtype=dtype,
        )
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )

    _assert_result_equal(res_out, ref_out, required_device=required_device)


# Negative rows run in both modes: invalid arguments must be rejected before any
# allocation is attempted.
_INVALID_ROWS = [
    ([2, 3], [1], {}),  # size/stride rank mismatch
    (3, [1], {}),  # size is not a sequence
    ([2.5, 3], [3, 1], {}),  # non-integer extent
    ([-2, 3], [3, 1], {}),  # negative extent
    pytest.param(
        [2, 3],
        [-1, 2],
        {},
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend",
            reason="Ascend native new_empty_strided accepts negative strides",
        ),
    ),
    pytest.param(
        [2, 3],
        [1 << 62, 1],
        {},
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "ascend",
            reason="Ascend native new_empty_strided accepts overflowing strides",
        ),
    ),
    ([2, 3], [3, 1], {"layout": torch.sparse_coo}),  # no sparse allocation backend
]
if torch.device(flag_gems.device).type != "cpu":
    # pin_memory is a host-memory contract: a non-CPU source is rejected. On a
    # CPU-configured build the same call is valid and is covered positively above.
    _INVALID_ROWS.append(([2, 3], [3, 1], {"pin_memory": True}))


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("size,stride,kwargs", _INVALID_ROWS)
def test_new_empty_strided_rejects_invalid_size_or_stride(size, stride, kwargs):
    if flag_gems.vendor_name == "kunlunxin" and kwargs.get("pin_memory") is True:
        pytest.skip(
            "Kunlunxin native new_empty_strided accepts and ignores device pin_memory=True"
        )
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError, ValueError, NotImplementedError)):
        flag_gems.new_empty_strided(inp, size, stride, **kwargs)


@pytest.mark.new_empty_strided
@pytest.mark.parametrize("self_operand", [3.14, [[1.0, 2.0]]], ids=["scalar", "list"])
def test_new_empty_strided_rejects_non_tensor_self(self_operand):
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.new_empty_strided(self_operand, [2], [1])


@pytest.mark.new_empty_strided
def test_new_empty_strided_out_rejects_non_tensor_buffer():
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.new_empty_strided(inp, [2, 3], [3, 1], out=[1, 2])


@pytest.mark.new_empty_strided
def test_new_empty_strided_out_rejects_dtype_override():
    # The out schema takes no dtype argument: the buffer's dtype is authoritative.
    inp = tu.make_input(torch.float32, (3, 4), ["-1", "1"])
    buf = torch.empty(2, 3, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.new_empty_strided(inp, [2, 3], [3, 1], out=buf, dtype=torch.float64)
