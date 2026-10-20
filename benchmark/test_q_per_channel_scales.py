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

import platform
import time
from dataclasses import asdict, dataclass

import flag_gems
import pytest
import torch
from torch.utils.benchmark import Timer

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# q_per_channel_scales returns stored per-channel quantizer metadata, so the default
# latency metrics describe the workload (byte throughput and FLOPs do not), the "dtype"
# axis is the quantized storage dtype, and every storage dtype runs through each
# per-channel qparams family it supports. Both call forms are measured.
#
# The per-channel axis is a schema parameter, so axes are derived per shape to exercise
# the first, middle and last dimensions of the shared default shapes. Rank-0, 1-D,
# zero-channel and empty shapes are added through the shared set_more_shapes() hook; a
# caller-supplied shape file still wins through the shared loader.

STORAGE_BYTES = {
    torch.qint8: torch.int8,
    torch.quint8: torch.uint8,
    torch.qint32: torch.int32,
}

# Both 8-bit and qint32 storage support either quantizer family. Integer zero
# points select float64 scales; floating zero points select float32 scales.
FAMILIES_BY_STORAGE = {
    torch.qint8: ("affine", "float_qparams"),
    torch.quint8: ("affine", "float_qparams"),
    torch.qint32: ("affine", "float_qparams"),
}

QPARAM_DTYPES = {
    "affine": (torch.float64, torch.int64),
    "float_qparams": (torch.float32, torch.float32),
}

RESULT_DTYPE = {"affine": torch.float64, "float_qparams": torch.float32}

STORAGE_DTYPES = [torch.qint8, torch.quint8, torch.qint32]

# Extra per-channel boundaries beyond the framework shape set: single channel, the
# remaining spec shapes, rank-0, empty storage, a zero-length channel axis and an empty
# tensor whose metadata is non-empty.
EXTRA_SHAPES = [
    (256,),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
    (),
    (0,),
    (4, 0, 3),
    (0, 5),
]

# (0, 5) has no channels on axis 0 but five on axis -1.
AXES_BY_SHAPE = {(0, 5): (0, -1)}


def _channels(shape, axis):
    # A rank-0 quantized tensor reports a single scale on axis 0.
    return 1 if len(shape) == 0 else shape[axis]


def _axes_for(shape):
    if len(shape) <= 1:
        return AXES_BY_SHAPE.get(shape, (0,))
    return AXES_BY_SHAPE.get(shape, tuple(sorted({0, len(shape) // 2, len(shape) - 1})))


def _quantized_input(shape, axis, storage_dtype, family, device):
    scales_dtype, zero_points_dtype = QPARAM_DTYPES[family]
    channels = _channels(shape, axis)
    scales = torch.arange(1, channels + 1, dtype=scales_dtype, device=device) * 0.5
    zero_points = (torch.arange(channels, dtype=torch.int32, device=device) % 7).to(
        zero_points_dtype
    )
    numel = 1
    for extent in shape:
        numel *= extent
    storage = (
        (torch.arange(numel, dtype=torch.int32, device=device) % 100)
        .to(STORAGE_BYTES[storage_dtype])
        .reshape(shape)
    )
    return torch.ops.aten._make_per_channel_quantized_tensor(
        storage, scales, zero_points, axis
    )


def _case_fn(shape, dtype):
    # Listing stays tensor-free: only JSON-compatible metadata is emitted here.
    shape = tuple(shape)
    for axis in _axes_for(shape):
        channels = _channels(shape, axis)
        for family in FAMILIES_BY_STORAGE[dtype]:
            yield base.BenchmarkCasePlan(
                shape={"input": shape},
                params={
                    "axis": axis,
                    "channels": channels,
                    "family": family,
                    "storage_dtype": str(dtype),
                },
                builder_args=(shape, axis, dtype, family),
            )


def _build_inputs_fn(plan, dtype, device):
    device = "cpu"  # The native input ABI is a CPU quantized tensor.
    shape, axis, storage_dtype, family = plan.builder_args
    return (_quantized_input(shape, axis, storage_dtype, family, device),)


def _build_inputs_fn_out(plan, dtype, device):
    device = "cpu"  # The native input ABI is a CPU quantized tensor.
    shape, axis, storage_dtype, family = plan.builder_args
    inp = _quantized_input(shape, axis, storage_dtype, family, device)
    # Allocation happens in the builder and is not timed. out is keyword-only in the
    # native schema, so it travels in the kwargs dict after the positional arguments.
    out_buf = torch.empty(
        _channels(shape, axis), dtype=RESULT_DTYPE[family], device=device
    )
    return inp, {"out": out_buf}


@dataclass
class _HostMetrics(base.BenchmarkMetrics):
    measurement_source: dict | None = None


class QPerChannelScalesBenchmark(OperatorBenchmark):
    def get_latency(self, op, *args, **kwargs):
        if self.is_backward:
            raise ValueError("Quantizer metadata access has no backward benchmark")
        if (
            not base.Config.reference_only
            and base.Config.mode != consts.BenchMode.OPERATOR
        ):
            raise ValueError("CPU host timing requires --mode operator")
        # GPU events cannot measure this host metadata API. One CPU thread and
        # repeated wall-clock samples keep tiny accessor timings meaningful.
        self._host_source = {
            "kind": "cpu_host",
            "actual_timing_mode": "host_wall_time",
            "device": "cpu",
            "architecture": platform.machine(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch": str(torch.__version__),
            "threads": 1,
            "timer": "torch.utils.benchmark.Timer",
            "scope": "CPU dispatch and quantizer metadata access",
        }
        latency = (
            Timer(
                stmt="op(*args, **kwargs)",
                globals={"op": op, "args": args, "kwargs": kwargs},
                num_threads=1,
                timer=time.perf_counter,
            )
            .blocked_autorange(min_run_time=0.05)
            .median
            * 1000
        )
        if base.Config.reference_only:
            active = [
                record
                for record in base.Config.reference_records
                if record.get("operator") == self.op_name
                and record.get("status") == "FAILED"
                and record.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise RuntimeError("Cannot identify the active CPU reference case")
            active[0]["measurement_source"] = self._host_source
        return latency

    def _measure_input(self, input, case_id=None):
        if "tflops" in self.to_bench_metrics or "gbps" in self.to_bench_metrics:
            raise ValueError("CPU metadata access supports latency metrics only")
        metric = super()._measure_input(input, case_id)
        if metric.speedup is not None:
            metric.speedup = metric.latency_base / metric.latency
        return _HostMetrics(**asdict(metric), measurement_source=self._host_source)

    def set_more_shapes(self):
        # Extra per-channel boundaries merged by the shared loader at the comprehensive
        # level; a caller shape file still overrides the default shapes.
        return super().set_more_shapes() + list(EXTRA_SHAPES)


@pytest.mark.q_per_channel_scales
def test_q_per_channel_scales():
    bench = QPerChannelScalesBenchmark(
        op_name="q_per_channel_scales",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.q_per_channel_scales,
        gems_op=getattr(flag_gems, "q_per_channel_scales", None),
        dtypes=STORAGE_DTYPES,
    )
    bench.run()


@pytest.mark.q_per_channel_scales
def test_q_per_channel_scales_out():
    bench = QPerChannelScalesBenchmark(
        op_name="q_per_channel_scales",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten.q_per_channel_scales.out,
        gems_op=getattr(flag_gems, "q_per_channel_scales", None),
        dtypes=STORAGE_DTYPES,
    )
    bench.run()
