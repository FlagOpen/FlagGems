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

# aten::is_contiguous reads size/stride/storage_offset only, so it has no
# kernel, no broadcast operand and no backward, and element values never change
# the verdict. Coverage is layout state (views, offsets, overlapping strides,
# empty and singleton tensors, lazy conj/neg bits) plus the memory_format
# overload, with the 5-range x 7-shape grid labelling the same layouts.

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


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
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
    cpu_encoding = flag_gems.vendor_name in ("ascend", "kunlunxin") and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    cpu_encoding = cpu_encoding or (
        flag_gems.vendor_name == "ascend" and dtype == torch.float64
    )
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


_KUNLUN_FP64_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="Kunlunxin narrows float64 storage to float32",
)


def _input_marks(dtype, shape, kind="plain"):
    return (_KUNLUN_FP64_SKIP,) if dtype == torch.float64 else ()


def _reference_layout(inp):
    # This predicate only reads geometry. A device copy can compact views and
    # resolve lazy flags; reconstruct the same shape/strides on the CPU instead.
    if utils.TO_CPU:
        return torch.empty_strided(
            inp.shape, inp.stride(), dtype=inp.dtype, device="cpu"
        )
    return inp


SUPPORTED_DTYPES = list(tu.REQUIRED_DTYPES) + [
    torch.float64,
    torch.bool,
    torch.complex64,  # exercises the lazy conjugate-bit rows
]

# (label, kind, base shape). All rows are metadata-cheap, so both modes collect
# them; quick mode only shrinks the value-range/shape grid and drops the
# positive special-value rows.
_LAYOUT_ROWS = [
    ("plain_2d", "plain", (4, 6)),
    ("scalar", "plain", ()),
    ("empty_1d", "plain", (0,)),
    ("empty_2d", "plain", (0, 3)),
    ("empty_transposed", "transposed", (0, 3)),
    ("transposed", "transposed", (4, 6)),
    ("transposed_singleton", "transposed", (1, 6)),
    ("row_step", "row_step", (8, 6)),
    ("column_step", "column_step", (4, 12)),
    ("offset_slice", "offset_slice", (8, 12)),
    ("unit_offset_1d", "unit_offset_1d", (16,)),
    ("unit_row", "unit_row", (8, 12)),
    ("expanded", "expanded", (1, 1)),
    ("expanded_1d", "expanded_1d", (1,)),
    ("expanded_singleton", "expanded_singleton", (1, 1)),
    ("channels_last", "channels_last", (2, 3, 4, 5)),
    ("channels_last_3d", "channels_last_3d", (2, 3, 4, 5, 6)),
    ("transposed_3d", "transposed_3d", (2, 3, 4)),
    ("transposed_4d", "transposed_4d", (2, 3, 4, 5)),
    ("singleton_trailing", "plain", (1, 4, 1)),
    ("singleton_leading", "plain", (1, 4, 6)),
    ("conj", "conj", (4, 6)),
    ("conj_transposed", "conj_transposed", (4, 6)),
    ("neg_view", "neg_view", (4, 6)),
]

# Queries accept every rank; an incompatible channels-last rank returns False.
# Only constructing a channels-last allocation requires a matching rank.
_MEMORY_FORMAT_ROWS = [
    (f"{label}_{memory_format}", kind, shape, memory_format)
    for label, kind, shape in _LAYOUT_ROWS
    for memory_format in (
        torch.contiguous_format,
        torch.preserve_format,
        torch.channels_last,
        torch.channels_last_3d,
    )
]

# Positive special-value rows are default-only.
_SPECIAL_VALUE_CASES = tu.selected_cases(
    [
        (dtype, scenario, kind)
        for dtype, scenario in tu.special_value_cases(SUPPORTED_DTYPES)
        for kind in ("plain", "step")
    ],
    quick=[],
)


def _input_state(inp):
    return (
        inp.dtype,
        inp.device,
        tuple(inp.shape),
        inp.stride(),
        inp.storage_offset(),
        inp.data_ptr(),
        inp.untyped_storage().data_ptr(),
        inp.requires_grad,
        inp.is_conj(),
        inp.is_neg(),
    )


def _make_layout_input(kind, shape, dtype):
    # Values are decorative for this predicate; only the layout matters.
    inp = _make_input(dtype, shape, ["-1", "1"])
    if kind == "plain":
        return inp
    if kind in ("transposed", "transposed_singleton", "empty_transposed"):
        return inp.t()
    if kind == "row_step":
        return inp[::2]
    if kind == "column_step":
        return inp[:, ::2]
    if kind == "offset_slice":
        return inp[2:6, 1:5]
    if kind == "unit_offset_1d":
        return inp[5:]
    if kind == "unit_row":
        return inp[3]
    if kind == "expanded":
        return inp.expand(3, 5)
    if kind == "expanded_1d":
        return inp.expand(7)
    if kind == "expanded_singleton":
        return inp.expand(1, 1)
    if kind == "channels_last":
        if flag_gems.vendor_name == "ascend" or (
            flag_gems.vendor_name == "kunlunxin"
            and dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        ):
            cpu = _copy_to(inp, "cpu").to(memory_format=torch.channels_last)
            return _copy_to(cpu, inp.device)
        return inp.to(memory_format=torch.channels_last)
    if kind == "channels_last_3d":
        if flag_gems.vendor_name == "ascend" or (
            flag_gems.vendor_name == "kunlunxin"
            and dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        ):
            cpu = _copy_to(inp, "cpu").to(memory_format=torch.channels_last_3d)
            return _copy_to(cpu, inp.device)
        return inp.to(memory_format=torch.channels_last_3d)
    if kind == "transposed_3d":
        return inp.transpose(0, 2)
    if kind == "transposed_4d":
        return inp.transpose(2, 3)
    if kind == "conj":
        return inp.conj()
    if kind == "conj_transposed":
        return inp.t().conj()
    if kind == "neg_view":
        # Lazy negation bit, set without an arithmetic op.
        return torch._neg_view(inp)
    raise ValueError(f"unknown layout kind: {kind}")


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
    + [((20, 320, 15), dtype) for dtype in SUPPORTED_DTYPES],
    quick=[((2, 19, 7), dtype) for dtype in SUPPORTED_DTYPES],
)


@pytest.mark.is_contiguous
@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(shape, dtype, marks=_input_marks(dtype, shape))
        for shape, dtype in _SHAPE_DTYPE_CASES
    ],
)
@pytest.mark.parametrize("value_range", tu.selected_ranges()[:1])
def test_is_contiguous(shape, value_range, dtype):
    inp = _make_input(dtype, shape, value_range)
    ref = torch.ops.aten.is_contiguous(_reference_layout(inp))

    res = flag_gems.is_contiguous(inp)

    assert type(res) is bool
    assert res == ref


@pytest.mark.is_contiguous
@pytest.mark.parametrize(
    "kind,shape,dtype",
    [
        pytest.param(
            kind,
            shape,
            dtype,
            marks=_input_marks(dtype, shape, kind),
            id=f"{label}-dtype{index}",
        )
        for label, kind, shape in _LAYOUT_ROWS
        for index, dtype in enumerate(SUPPORTED_DTYPES)
    ],
)
def test_is_contiguous_layout(kind, shape, dtype):
    inp = _make_layout_input(kind, shape, dtype)
    ref = torch.ops.aten.is_contiguous(_reference_layout(inp))

    res = flag_gems.is_contiguous(inp)

    assert type(res) is bool
    assert res == ref


@pytest.mark.is_contiguous
@pytest.mark.parametrize(
    "kind,shape,memory_format,dtype",
    [
        pytest.param(
            kind,
            shape,
            memory_format,
            dtype,
            marks=_input_marks(dtype, shape, kind),
            id=f"{label}-dtype{index}",
        )
        for label, kind, shape, memory_format in _MEMORY_FORMAT_ROWS
        for index, dtype in enumerate(SUPPORTED_DTYPES)
    ],
)
def test_is_contiguous_memory_format(kind, shape, memory_format, dtype):
    inp = _make_layout_input(kind, shape, dtype)
    ref = torch.ops.aten.is_contiguous.memory_format(
        _reference_layout(inp), memory_format
    )

    res = flag_gems.is_contiguous(inp, memory_format)

    assert type(res) is bool
    assert res == ref


@pytest.mark.is_contiguous
@pytest.mark.parametrize(
    "dtype,scenario,kind",
    [
        pytest.param(dtype, scenario, kind, marks=_input_marks(dtype, (5,)))
        for dtype, scenario, kind in _SPECIAL_VALUE_CASES
    ],
)
def test_is_contiguous_special_values(dtype, scenario, kind):
    payload = _make_special_input(dtype, scenario)
    inp = payload if kind == "plain" else payload[::2]
    ref = torch.ops.aten.is_contiguous(_reference_layout(inp))

    res = flag_gems.is_contiguous(inp)

    assert type(res) is bool
    assert res == ref


@pytest.mark.is_contiguous
def test_is_contiguous_keeps_input_state():
    inp = _make_input(torch.float32, (8, 12), ["-1", "1"])[:, ::2]
    ref = torch.ops.aten.is_contiguous(_reference_layout(inp))
    state_before = _input_state(inp)

    res = flag_gems.is_contiguous(inp)

    assert _input_state(inp) == state_before
    assert type(res) is bool
    assert res == ref


@pytest.mark.is_contiguous
def test_is_contiguous_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_contiguous("not a tensor")


@pytest.mark.is_contiguous
def test_is_contiguous_rejects_extra_argument():
    inp = _make_input(torch.float32, (4, 6), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_contiguous(inp, torch.contiguous_format, 1)


@pytest.mark.is_contiguous
@pytest.mark.parametrize("memory_format", [None, "x", 1.5, torch.float32])
def test_is_contiguous_rejects_invalid_memory_format(memory_format):
    # Unknown integer memory_format codes are accepted natively, so only
    # non-integer values are negative cases.
    inp = _make_input(torch.float32, (4, 6), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_contiguous(inp, memory_format)
