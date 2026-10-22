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

"""Benchmark for ``aten::empty_strided``.

A case is a (size, stride, dtype) triple. The shared default/comprehensive
shapes provide the sizes and the extra stride layouts are derived from each of
them. Planning only touches metadata, so ``--list-cases`` and ``--case-id``
replay allocate nothing.
"""

import pytest
import torch

import flag_gems

from . import base, consts

EMPTY_STRIDED_LAYOUTS = [
    ((), ()),
    ((256,), (1,)),
    ((1024, 1024), (1024, 1)),
    ((1024, 1024), (1, 1024)),
    ((20, 320, 15), (4800, 15, 1)),
    ((16, 128, 64, 60), (491520, 3840, 60, 1)),
    ((512, 512), (1024, 2)),
    ((7, 13, 5), (65, 5, 1)),
    ((0, 3), (3, 1)),
    ((2, 3), (1, 1)),
    ((4 * 1024 * 1024,), (1,)),
]

_LAYOUTS = ("contiguous", "reversed", "padded", "overlap", "zero_stride")


def _contiguous_strides(size):
    strides = [0] * len(size)
    running = 1
    for axis in range(len(size) - 1, -1, -1):
        strides[axis] = running
        running *= size[axis]
    return tuple(strides)


def _strides_for(size, layout):
    contiguous = _contiguous_strides(size)
    if layout == "reversed":
        return tuple(reversed(contiguous))
    if layout == "padded":
        # One padding element per axis: a storage larger than numel.
        return tuple(step * 2 for step in contiguous)
    if layout == "zero_stride":
        return tuple(0 for _ in contiguous)
    if layout == "overlap":
        # Every axis advances by one element: an overlapping, smaller storage.
        return tuple(1 for _ in contiguous)
    return contiguous


# Every element type the allocator accepts on this backend (native probe: all
# allocate). consts.FP8_DTYPES is not iterated directly because it holds None on
# backends without fp8 support.
_ALLOCATOR_DTYPES = list(
    dict.fromkeys(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [torch.float64, torch.complex32, torch.complex128]
        + [
            dtype
            for dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            if flag_gems.runtime.device.support_fp8
        ]
    )
)


def _case_fn(shape, dtype):
    del dtype
    if len(shape) == 2 and isinstance(shape[0], (tuple, list)):
        size, stride = map(tuple, shape)
        yield base.BenchmarkCasePlan(
            shape={"size": list(size), "stride": list(stride)},
            params={"layout": "explicit"},
            builder_args=(size, stride),
        )
        return
    size = tuple(shape)
    seen = set()
    for layout in _LAYOUTS:
        stride = _strides_for(size, layout)
        if stride in seen:
            continue
        seen.add(stride)
        yield base.BenchmarkCasePlan(
            shape={"size": list(size), "stride": list(stride)},
            params={"layout": layout},
            builder_args=(size, stride),
        )


def _build_inputs_fn(plan, dtype, device):
    size, stride = plan.builder_args
    # Positional size/stride plus keyword dtype/device.
    return list(size), list(stride), {"dtype": dtype, "device": device}


class EmptyStridedBenchmark(base.GenericBenchmark):
    # These backends execute one original plan per pytest item, allowing a real
    # resource skip to finish only that plan. Keep the original catalog IDs.
    canonical_nodeid = None
    selected_case = None

    def _case_id(self, dtype, ordinal):
        if self.canonical_nodeid is None:
            return super()._case_id(dtype, ordinal)
        return (
            f"{self.canonical_nodeid}::{base.Config.bench_level.value}::"
            f"{str(dtype).removeprefix('torch.')}::{ordinal}"
        )

    def get_case_iter(self, dtype):
        if self.selected_case is None:
            yield from super().get_case_iter(dtype)
        elif self.selected_case.dtype == dtype:
            yield self.selected_case

    def build_inputs(self, case):
        if flag_gems.vendor_name == "kunlunxin" and case.dtype in (
            torch.float64,
            torch.complex128,
        ):
            pytest.skip(
                "Kunlunxin empty_strided narrows requested float64/complex128 "
                "to float32/complex64 on the target device"
            )
        size, stride = case.builder_args[0].builder_args
        # Only these two original core plans require 32 GiB minus 16 bytes.
        # Native context memory prevents them fitting the measured 32 GiB
        # Iluvatar cards. Do not turn unrelated transient load into more skips.
        capacity_cases = (
            ((1073741824,), (2,)),
            ((1024, 1024, 1024), (2097152, 2048, 2)),
        )
        if (
            flag_gems.vendor_name == "iluvatar"
            and case.dtype == torch.complex128
            and (tuple(size), tuple(stride)) in capacity_cases
            and torch.cuda.get_device_properties(self.device).total_memory <= 2**35
        ):
            elements = 1 + sum((dim - 1) * step for dim, step in zip(size, stride))
            required = elements * case.dtype.itemsize
            # Driver-free memory excludes reusable allocator reservations.
            # Release this process's unused cache before the capacity check.
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            if required > free:
                pytest.skip(
                    f"Iluvatar empty_strided allocation needs {required} bytes "
                    f"but only {free} device bytes are free after cache release; "
                    f"size={tuple(size)}, stride={tuple(stride)}, dtype={case.dtype}"
                )
        return super().build_inputs(case)

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(self.shapes) + EMPTY_STRIDED_LAYOUTS


def _make_benchmark():
    return EmptyStridedBenchmark(
        op_name="empty_strided",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.empty_strided,
        gems_op=getattr(flag_gems, "empty_strided", None),
        dtypes=_ALLOCATOR_DTYPES,
    )


@pytest.fixture
def _empty_strided_case():
    return None


def pytest_generate_tests(metafunc):
    if (
        metafunc.function.__name__ != "test_empty_strided"
        or flag_gems.vendor_name not in ("iluvatar", "kunlunxin")
    ):
        return
    canonical_nodeid = metafunc.definition.nodeid
    if base.Config.list_cases or base.Config.query:
        selected = [(canonical_nodeid, None)]
        labels = ["catalog"]
    else:
        planner = _make_benchmark()
        planner.canonical_nodeid = canonical_nodeid
        cases = planner.list_cases().cases
        selected = [(canonical_nodeid, case) for case in cases]
        labels = [
            f"{str(case.dtype).removeprefix('torch.')}::{case.ordinal}"
            for case in cases
        ]
    metafunc.parametrize("_empty_strided_case", selected, ids=labels)


@pytest.mark.empty_strided
def test_empty_strided(_empty_strided_case):
    bench = _make_benchmark()
    if _empty_strided_case is not None:
        bench.canonical_nodeid, bench.selected_case = _empty_strided_case
    bench.run()
