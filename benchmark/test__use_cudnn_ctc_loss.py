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

"""Benchmark for ``aten::_use_cudnn_ctc_loss``.

The op is a boolean capability predicate: it inspects rank, dtype, device and
the two length lists, then returns a Python bool without allocating anything, so
what is measured is dispatch + predicate cost for the accepted float32 / rank-3
log_probs with CPU int32 targets, in both native call forms (Python ``int[]``
lengths and tensor lengths). A rejected dtype does not open a second code path,
so only float32 log_probs is timed.

``torch_op`` is the ATen reference (the perf comparison baseline) and ``gems_op``
is the FlagGems candidate resolved through ``--override``; both receive the same
positional operands and the same ``blank`` keyword. Cases come from ``case_fn``
only, so ``--list-cases`` and ``--case-id`` replay allocate no tensors and call no
operator, and the plans used for listing are the same ones used for execution.
"""

import platform
import time
from dataclasses import asdict, dataclass

import flag_gems
import pytest
import torch
from torch.utils.benchmark import Timer

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# Rank-3 (T, N, C) log_probs: every input_length equals T and every target length
# stays below 256, which keeps the predicate on its accepted path.
CTC_SHAPES = [
    (128, 8, 16),
    (256, 16, 32),
    (512, 32, 64),
    (1024, 64, 128),
    (2048, 8, 64),
]

_TARGET_LENGTH = 8
_LENGTHS_FORMS = ("list", "tensor")


def _case_fn(shape, dtype):
    del dtype
    log_probs_shape = tuple(shape)
    batch = log_probs_shape[1] if len(log_probs_shape) == 3 else 1
    time_extent = log_probs_shape[0] if log_probs_shape else 0
    input_lengths = [time_extent] * batch
    target_lengths = [_TARGET_LENGTH] * batch
    for form in _LENGTHS_FORMS:
        yield base.BenchmarkCasePlan(
            shape={
                "log_probs": list(log_probs_shape),
                "targets": [batch * _TARGET_LENGTH],
            },
            params={"blank": 0, "lengths_form": form},
            builder_args=(log_probs_shape, input_lengths, target_lengths, form),
        )


def _build_inputs_fn(plan, dtype, device):
    log_probs_shape, input_lengths, target_lengths, form = plan.builder_args
    log_probs = utils.generate_tensor_input(log_probs_shape, dtype, device)
    # targets must be rank-1 int32 and, for the int[] form, on CPU: the native
    # predicate tests targets.device().type() == at::kCPU.
    targets = torch.zeros(
        len(target_lengths) * _TARGET_LENGTH, dtype=torch.int32, device="cpu"
    )
    if form == "tensor":
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32, device=device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32, device=device)
    # unpack_to_args_kwargs keeps tensors/lists positional and merges a trailing
    # dict into the call kwargs, so the operands are returned flat -- a nested
    # args tuple would be passed as a single operand.
    return (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        {"blank": plan.params["blank"]},
    )


def _reference(log_probs, targets, input_lengths, target_lengths, blank):
    # This is the eligibility predicate, not CTC computation. Non-CUDA input
    # cannot use cuDNN; no CPU dispatch of the CUDA-only native API is needed.
    return False


@dataclass
class _HostMetrics(base.BenchmarkMetrics):
    measurement_source: dict | None = None


class UseCudnnCtcLossBenchmark(OperatorBenchmark):
    def get_latency(self, op, *args, **kwargs):
        if self.is_backward:
            raise ValueError("A capability predicate has no backward benchmark")
        if (
            not base.Config.reference_only
            and base.Config.mode != consts.BenchMode.OPERATOR
        ):
            raise ValueError("CPU host timing requires --mode operator")
        # GPU events cannot measure this host metadata API. One CPU thread and
        # repeated wall-clock samples keep tiny accessor timings meaningful.
        self._host_source = {
            "kind": (
                "cuda_dispatch_host_predicate"
                if torch.device(flag_gems.device).type == "cuda"
                else "portable_host_predicate"
            ),
            "actual_timing_mode": "host_wall_time",
            "device": "cpu",
            "architecture": platform.machine(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch": str(torch.__version__),
            "threads": 1,
            "timer": "torch.utils.benchmark.Timer",
            "scope": "Host capability predicate; no CTC loss computation",
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
            raise ValueError("A host predicate supports latency metrics only")
        metric = super()._measure_input(input, case_id)
        if metric.speedup is not None:
            metric.speedup = metric.latency_base / metric.latency
        return _HostMetrics(**asdict(metric), measurement_source=self._host_source)

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Other ranks are valid predicate workloads returning False. Retain
        # their requested shapes alongside the rank-3 cuDNN eligibility cases.
        self.shapes = list(
            dict.fromkeys([tuple(shape) for shape in self.shapes] + CTC_SHAPES)
        )


@pytest.mark.use_cudnn_ctc_loss
def test__use_cudnn_ctc_loss():
    bench = UseCudnnCtcLossBenchmark(
        op_name="_use_cudnn_ctc_loss",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=(
            torch.ops.aten._use_cudnn_ctc_loss
            if torch.device(flag_gems.device).type == "cuda"
            else _reference
        ),
        gems_op=getattr(flag_gems, "_use_cudnn_ctc_loss", None),
        dtypes=[torch.float32],
    )
    bench.run()
