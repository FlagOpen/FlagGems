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


"""Performance benchmark for aten::cudnn_grid_sampler.

The operator samples a rank-4 input at rank-4 grid coordinates, so each case is
an (input_shape, grid_shape) pair with ``grid.shape[-1] == 2`` and equal batch
sizes.

FLAGGEMS_NV_BASELINE explicitly selects a local ops-baseline JSON. Only the
matched NVIDIA latency is imported; candidates still run on the target, and
records retain both device identities for the raw cross-chip ratio.
"""

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

_HAS_FP64 = bool(getattr(flag_gems.runtime.device, "support_fp64", False))
_DTYPES = [torch.float32, torch.float16] + ([torch.float64] if _HAS_FP64 else [])

# (input_shape, grid_shape) pairs; grid last dim must stay 2.
_BENCH_CASES = [
    ((16, 128, 64, 60), (16, 8, 8, 2)),
    ((1, 1, 1024, 1024), (1, 32, 32, 2)),
    ((8, 16, 128, 128), (8, 32, 32, 2)),
    ((4, 3, 256, 256), (4, 64, 64, 2)),
    ((2, 3, 512, 512), (2, 16, 16, 2)),
    ((2, 6, 128, 128), (2, 16, 16, 2)),
]


def _case_fn(case, dtype):
    # set_shapes feeds one (input_shape, grid_shape) descriptor per case.
    del dtype
    input_shape, grid_shape = case
    yield base.BenchmarkCasePlan(
        shape={"input": list(input_shape), "grid": list(grid_shape)},
        params={"samples": list(grid_shape[:3])},
        builder_args=(input_shape, grid_shape),
    )


def _build_inputs_fn(plan, dtype, device):
    input_shape, grid_shape = plan.builder_args
    inp = torch.randn(input_shape, dtype=dtype, device=device)
    # Measured native behaviour: coordinates outside [-1, 1] fall back to the
    # zero padding, so a [-1, 1] grid keeps the timed samples on real pixels.
    grid = torch.rand(grid_shape, dtype=dtype, device=device) * 2 - 1
    return inp, grid


@dataclass
class _CrossMachineMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class CudnnGridSamplerBenchmark(OperatorBenchmark):
    """Two-phase benchmark over rank-4 (input, grid) pairs of growing size."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)

    def set_more_shapes(self):
        # The level-2 extras are plain 1-D/2-D/3-D shapes and cannot express the
        # (input, grid) pair this operator needs, so listing and execution use
        # exactly _BENCH_CASES at every bench level.
        return []

    def _external_input_config(self, args):
        if len(args) != 2 or any(t.layout != torch.strided for t in args):
            raise ValueError("cuDNN baseline requires two dense input/grid tensors")
        if args[0].device != args[1].device:
            raise ValueError("Input and grid devices must match")
        return {
            "inputs": {
                "args": [
                    {
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "stride": list(t.stride()),
                        "requires_grad": t.requires_grad,
                    }
                    for t in args
                ],
                "kwargs": {},
            },
            "source_commit": "9d57a23b4386bd669152a935fe88a96db907da7c",
            "benchmark": "benchmark/test_cudnn_grid_sampler.py",
            "descriptor": [list(t.shape) for t in args],
            "scope": "core",
            "samples": list(args[1].shape[:3]),
            "variant": 0,
        }

    @cached_property
    def _nvidia_baseline(self):
        # Explicit local input; listing/preflight/profile never load this file.
        path = Path(os.environ["FLAGGEMS_NV_BASELINE"]).resolve(strict=True)
        payload = path.read_bytes()

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate baseline JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"Non-finite baseline JSON constant: {value}")

        data = json.loads(
            payload, object_pairs_hook=unique_keys, parse_constant=reject_constant
        )
        task = data["_collection"]["tasks"][self.op_name]
        environment = data["_environments"][str(task["device"])]
        commit = environment["source"]["git_commit"]
        if (
            data[self.op_name]["native_api"] != f"ops.{self.op_name}"
            or task["status"] != "completed"
            or task["returncode"] != 0
            or environment["run"]["repo"] != "pytorch"
            or not environment["hardware"]["gpu"]["name"].startswith("NVIDIA ")
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
            or not environment["software"]["packages"]["torch"]
            or environment["source"]["git_dirty"] is not False
        ):
            raise ValueError(
                "External baseline needs a completed, traceable NVIDIA run"
            )
        return data[self.op_name]["shapes"], {
            "kind": "external_nvidia",
            "artifact": str(path),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "environment": environment,
            "ratio": "raw NVIDIA latency / target latency; no peak normalization",
            "target_timing_mode": base.Config.mode.value,
        }

    def _import_nvidia_latency(self, args, kwargs):
        if kwargs or self.is_backward or "tflops" in self.to_bench_metrics:
            raise ValueError(
                "External baseline supports these forward positional cases only"
            )
        config = self._external_input_config(args)
        shapes, provenance = self._nvidia_baseline
        matches = [
            record
            for dtypes in shapes.values()
            for dtype, record in dtypes.items()
            if dtype == str(args[0].dtype) and record.get("config") == config
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one NVIDIA record matching the actual inputs"
            )
        record = matches[0]
        latency_us = record.get("T_us")
        if (
            isinstance(latency_us, bool)
            or not isinstance(latency_us, (int, float))
            or not math.isfinite(latency_us)
            or latency_us <= 0
        ):
            raise ValueError("NVIDIA T_us must be finite and positive")
        latency_ms = latency_us / 1000
        recorded_ms = record.get("latency_ms", latency_ms)
        if (
            isinstance(recorded_ms, bool)
            or not isinstance(recorded_ms, (int, float))
            or not math.isfinite(recorded_ms)
            or not math.isclose(recorded_ms, latency_ms, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError("NVIDIA latency units disagree")
        self._imported_baseline = {
            **provenance,
            "config": config,
            "target_environment": {
                "vendor": flag_gems.vendor_name,
                "device": str(args[0].device),
                "device_name": base.torch_device_fn.get_device_name(),
                "torch": str(torch.__version__),
            },
        }
        if base.Config.reference_only:
            # The base runner creates all records before traversing inputs.
            active = [
                r
                for r in base.Config.reference_records
                if r.get("operator") == self.op_name
                and r.get("status") == "FAILED"
                and r.get("stage") == "benchmark_reference"
            ]
            if len(active) != 1:
                raise RuntimeError("Cannot identify the active external-reference case")
            active[0]["baseline_source"] = self._imported_baseline
        return latency_ms

    def get_latency(self, op, *args, **kwargs):
        importing = os.environ.get("FLAGGEMS_NV_BASELINE") and (
            base.Config.reference_only
            or getattr(self, "_native_measurement_pending", False)
        )
        if importing:
            if base.Config.skip_native or op is not self.torch_op:
                raise ValueError(
                    "External baseline requires the original reference measurement"
                )
            # Only the first, reference measurement imports. Even an identical
            # candidate callable must execute through the real timer next.
            self._native_measurement_pending = False
            return self._import_nvidia_latency(args, kwargs)
        return super().get_latency(op, *args, **kwargs)

    def _measure_input(self, input, case_id=None):
        self._imported_baseline = None
        self._native_measurement_pending = False
        external = bool(os.environ.get("FLAGGEMS_NV_BASELINE"))
        if external:
            if self.is_backward or "tflops" in self.to_bench_metrics:
                raise ValueError(
                    "External baseline supports forward latency measurements only"
                )
            if base.Config.skip_native:
                raise ValueError("Do not combine an external baseline with skip_native")
        self._native_measurement_pending = (
            external and "latency_base" in self.to_bench_metrics
        )
        try:
            metric = super()._measure_input(input, case_id)
            if external and metric.latency is not None:
                if (
                    isinstance(metric.latency, bool)
                    or not isinstance(metric.latency, (int, float))
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError("Target latency must be finite and positive")
            if self._imported_baseline is not None:
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                return _CrossMachineMetrics(
                    **asdict(metric), baseline_source=self._imported_baseline
                )
            return metric
        finally:
            self._native_measurement_pending = False

    def _run_reference_cases(self, case_ids):
        first_record = len(base.Config.reference_records)
        if os.environ.get("FLAGGEMS_NV_BASELINE") and base.Config.skip_native:
            raise ValueError("Do not combine an external baseline with skip_native")
        try:
            return super()._run_reference_cases(case_ids)
        finally:
            for record in base.Config.reference_records[first_record:]:
                if record.get("baseline_source") and record.get("status") == "PASSED":
                    record["stage"] = "external_baseline"


@pytest.mark.cudnn_grid_sampler
def test_cudnn_grid_sampler():
    bench = CudnnGridSamplerBenchmark(
        op_name="cudnn_grid_sampler",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.cudnn_grid_sampler,
        gems_op=getattr(flag_gems, "cudnn_grid_sampler", None),
        dtypes=_DTYPES,
    )
    bench.run()
