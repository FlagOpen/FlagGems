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

import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
import torch
from torch.utils.benchmark import Timer

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# aten::_nested_tensor_strides(Tensor self) -> Tensor reads the (num_tensors,
# num_dims) int64 strides metadata of a strided-layout nested tensor. Its cost is
# proportional to num_tensors * num_dims and independent of the stored values, so
# the benchmark uses (num_tensors, num_dims) layouts instead of the dense
# core_shapes set, which is dominated by huge 1-D tensors that are meaningless
# for a nested-tensor batch.
_NESTED_STRIDES_SHAPES = [
    (16, 2),
    (64, 3),
    (256, 3),
    (1024, 2),
    (2048, 4),
    (4096, 5),
    (8192, 3),
]


_HOST_CORE_SHAPES = _NESTED_STRIDES_SHAPES


def _input_contract(inp):
    assert isinstance(inp, torch.Tensor) and inp.is_nested
    parts = list(inp.unbind())
    return {
        "args": [
            {
                "is_nested": True,
                "layout": str(inp.layout),
                "dtype": str(inp.dtype),
                "device": "accelerator",
                "requires_grad": inp.requires_grad,
                "components": [
                    {
                        "shape": list(x.shape),
                        "stride": list(x.stride()),
                        "storage_offset": x.storage_offset(),
                    }
                    for x in parts
                ],
                "storage_elements": inp.values().numel(),
            }
        ],
        "kwargs": {},
    }


def _native_host_record(operator, case_id, inp):
    path = Path(os.environ["FLAGGEMS_NESTED_METADATA_H800_HOST_BASELINE"])
    raw = path.read_bytes()
    # Pin the actual published native H800 collection, not a portable oracle.
    expected_digest = "434e8ffca47e3acec7aeb9e4cf0f3cd0dd6f256475cff95a5bcfa138754b616f"
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("Unexpected native nested-metadata host artifact")
    data = json.loads(raw)
    task = data["_tasks"][operator]
    source = data["_source"]["flaggems"]
    if (
        data["_format"] != "flaggems_native_host_v1"
        or source["dirty"] is not False
        or source["commit"] != "ca1f2c01731fb5230e4b3ff9e9247e5a76139763"
        or task["status"] != "completed"
        or task["returncode"] != 0
    ):
        raise ValueError(
            "Native host collection did not complete at the expected source"
        )
    matches = [x for x in data["operators"][operator] if x["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError("Expected exactly one native host timing case")
    record = matches[0]
    measurement = record["host_measurement"]
    config = measurement["inputs"]
    # Source device number is a role; component geometry remains exact.
    expected = _input_contract(inp)
    config = {
        "args": [dict(config["args"][0], device="accelerator")],
        "kwargs": config["kwargs"],
    }
    if (
        config != expected
        or record["operator"] != operator
        or record["status"] != "PASSED"
        or measurement["kind"] != "cuda_dispatch_host_metadata"
        or measurement["synchronized_cuda"] is not False
        or measurement["native_api"]
        not in (f"aten.{operator}", f"aten.{operator}.default")
        or measurement["environment"]["timer"] != "torch.utils.benchmark.Timer"
        or "H800" not in measurement["environment"]["cuda_device"]
        or measurement["environment"]["threads"] != 1
        or measurement["sample_count"] < 1
        or measurement["number_per_run"] < 1
    ):
        raise ValueError(
            "Native host record differs from the current input or measurement contract"
        )
    latency = record["latency_ms"]
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency <= 0
    ):
        raise ValueError("Native host latency must be finite and positive")
    return latency, {
        "kind": "external_native_h800_host",
        "artifact_sha256": expected_digest,
        "source": data["_source"],
        "source_measurement": measurement,
        "case_id": case_id,
        "ratio": "source host latency / local candidate host latency",
    }


@dataclass
class _HostMetrics(base.BenchmarkMetrics):
    measurement_sources: dict | None = None


class _HostMetadataTiming:
    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name == "ascend" and op is self.torch_op:
            raise RuntimeError(
                "Ascend nested metadata benchmark requires an explicit candidate"
            )
        return op, dispatch, overridden

    def get_latency(self, op, *args, **kwargs):
        role = "reference" if op is self.torch_op else "candidate"
        if flag_gems.vendor_name == "ascend" and role == "reference":
            if kwargs or len(args) != 1:
                raise ValueError("Unexpected nested metadata call ABI")
            inp = args[0]
            num_tensors = len(inp.unbind())
            rank = inp.unbind()[0].dim()
            shape = (num_tensors, rank)
            if shape not in _HOST_CORE_SHAPES or inp.dtype not in consts.FLOAT_DTYPES:
                raise ValueError("Input is outside the native H800 host core grid")
            case_id = (
                f"benchmark/test_{self.op_name}.py::test_{self.op_name}::core::"
                f"{str(inp.dtype).removeprefix('torch.')}::{_HOST_CORE_SHAPES.index(shape)}"
            )
            latency, source = _native_host_record(self.op_name, case_id, inp)
        else:
            if (
                not base.Config.reference_only
                and base.Config.mode != consts.BenchMode.OPERATOR
            ):
                raise ValueError("Host metadata timing requires operator mode")
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
            source = {
                "kind": "local_host_wall_time",
                "timer": "torch.utils.benchmark.Timer",
                "threads": 1,
                "torch": str(torch.__version__),
                "platform": platform.platform(),
            }
        self._measurement_sources[role] = source
        return latency

    def _measure_input(self, input, case_id=None):
        if any(x in self.to_bench_metrics for x in ("tflops", "gbps")):
            raise ValueError("Host metadata supports latency metrics only")
        self._measurement_sources = {}
        metric = super()._measure_input(input, case_id)
        return _HostMetrics(
            **asdict(metric), measurement_sources=self._measurement_sources
        )


def _case_fn(shape, dtype):
    # One (num_tensors, num_dims) layout per Workload; the builder args carry the
    # layout so build_inputs_fn stays dtype-agnostic.
    del dtype
    num_tensors, num_dims = shape
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=((num_tensors, num_dims),),
    )


def _build_inputs_fn(plan, dtype, device):
    # build_inputs_fn is the second phase of the two-phase GenericBenchmark: it is
    # called per (case, dtype) with the parsed plan and the test device. Components
    # are ordinary contiguous tensors whose dim 0 is ragged (length in [1, 8]) and
    # whose remaining dims are fixed at 4; their values are irrelevant to the
    # operator, so plain randn input is used here (the correctness file is the one
    # that sweeps the spec value ranges).
    num_tensors, num_dims = plan.builder_args[0]
    gen = torch.Generator("cpu").manual_seed(0)
    lengths = torch.randint(1, 9, (num_tensors,), generator=gen, device="cpu").tolist()
    components = [
        torch.randn((length,) + (4,) * (num_dims - 1), dtype=dtype, device=device)
        for length in lengths
    ]
    return torch.nested.nested_tensor(components, device=device), {}


class NestedTensorStridesBenchmark(_HostMetadataTiming, OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to (num_tensors, num_dims) shapes.

    ``set_shapes``/``set_more_shapes`` are overridden because the default
    implementations read the dense core_shapes.yaml set, which has no meaning for
    a nested-tensor batch. ``record_shapes`` is overridden because a
    strided-layout NestedTensor does not support ``Tensor.size()`` (it raises
    "NestedTensorImpl doesn't support sizes"); the strides metadata tensor
    returned by the operator itself is used as the shape descriptor instead.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_NESTED_STRIDES_SHAPES)

    def set_more_shapes(self):
        return []

    def record_shapes(self, *args, **kwargs):
        def deep_parse(item):
            if isinstance(item, torch.Tensor):
                if item.is_nested:
                    parts = item.unbind()
                    return ("nested", len(parts), parts[0].dim() if parts else 0)
                return item.size()
            elif isinstance(item, (int, float, str, torch.dtype)):
                return item
            elif isinstance(item, (list, tuple)):
                return [deep_parse(sub_item) for sub_item in item]
            elif isinstance(item, dict):
                return {key: deep_parse(value) for key, value in item.items()}
            return None

        parsed_args = [deep_parse(arg) for arg in args]
        parsed_kwargs = {key: deep_parse(value) for key, value in kwargs.items()}
        if parsed_args and parsed_kwargs:
            return parsed_args, parsed_kwargs
        return parsed_args if parsed_args else parsed_kwargs


@pytest.mark.underscore_nested_tensor_strides
def test__nested_tensor_strides():
    # torch_op is the perf comparison reference; gems_op is the candidate under
    # test. They share the same single-argument call semantics
    # (Tensor) -> Tensor. The candidate name starts with an underscore, so the
    # default is fetched with getattr to stay importable before the kernel is
    # registered; KernelGen's injected candidate is visible here at call time.
    bench = NestedTensorStridesBenchmark(
        op_name="_nested_tensor_strides",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._nested_tensor_strides,
        gems_op=getattr(flag_gems, "_nested_tensor_strides", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
