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
from pathlib import Path
import time

import pytest
import torch
from torch.utils.benchmark import Timer

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# _lazy_clone relocates storage only. The shared shape file / default shapes are
# kept unchanged (no override, no cap); the extra original shapes
# below are contributed at the comprehensive level, and every case is expanded
# into a contiguous and a transposed plan so the copied non-contiguous geometry
# is measured as well.
LAZY_CLONE_EXTRA_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
]

# Static capability flags read at import time: listing allocates no tensor and
# calls no operator.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


# The operator is a dtype-agnostic storage relocation, so every supported dtype
# family is timed: float, complex, int and bool.
BENCH_DTYPES = [
    dtype
    for dtype in (
        consts.FLOAT_DTYPES
        + consts.COMPLEX_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + [torch.float64, torch.complex128, torch.float8_e4m3fn, torch.float8_e5m2]
    )
    if _dtype_supported(dtype)
]


_HOST_BASELINE_SHA256 = (
    "aef2602ae7dc95c3bf428aec2fc6779a20c4cce7a8a9620e0d9a2ada1383f1ff"
)


def _load_host_baseline():
    # Preparation supplies the exact ops-baseline PR26 artifact. Never download
    # data during pytest, silently time CPU in place of native, or invent zeros.
    path = os.environ.get("FLAGGEMS_LAZY_CLONE_HOST_BASELINE")
    if not path:
        raise RuntimeError(
            "Kunlunxin _lazy_clone requires FLAGGEMS_LAZY_CLONE_HOST_BASELINE "
            "pointing to ops-baseline/op_pytorch_h800_lazy_clone_host_baseline.json"
        )
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != _HOST_BASELINE_SHA256:
        raise ValueError("_lazy_clone H800 host baseline checksum mismatch")
    data = json.loads(payload)
    if (
        data.get("_format") != "ops_pytorch_lazy_clone_host_v1"
        or data.get("_status") != "complete"
        or data.get("operator") != "_lazy_clone"
        or data["_measurement"]["kind"] != "cuda_dispatch_host_metadata"
    ):
        raise ValueError("_lazy_clone baseline is not a complete native host artifact")
    records = {record["case_id"]: record for record in data["records"]}
    if len(data["records"]) != 60 or len(records) != 60:
        raise ValueError("_lazy_clone baseline must contain the original 60 cases")
    return records


def _transposed_shape(shape):
    return list(shape[:-2]) + [shape[-1], shape[-2]]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"layout": "contiguous"},
        builder_args=(shape, "contiguous"),
    )
    if len(shape) >= 2:
        yield base.BenchmarkCasePlan(
            shape={"input": _transposed_shape(shape), "storage": list(shape)},
            params={"layout": "transposed"},
            builder_args=(shape, "transposed"),
        )


def _build_inputs_fn(plan, dtype, device):
    storage_shape, layout = plan.builder_args
    inp = torch.empty(storage_shape, dtype=dtype, device=device)
    if layout == "transposed":
        inp = inp.transpose(-1, -2)
    # Flat positional arguments with a trailing kwargs dict.
    return inp, {}


class LazyCloneBenchmark(OperatorBenchmark):
    """Measure host metadata cost; replay native H800 on Kunlunxin.

    The injected candidate, including CPU-oracle transport, is timed locally.
    The ratio compares API call cost, not GPU copy bandwidth.
    """

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name == "kunlunxin" and op is self.torch_op:
            raise RuntimeError(
                "Kunlunxin _lazy_clone requires an explicit local candidate; "
                "native-reference fallback cannot be timed as the candidate"
            )
        return op, dispatch, overridden

    def _measure_input(self, input, case_id=None):
        self._native_host_case_id = case_id
        return super()._measure_input(input, case_id=case_id)

    def get_latency(self, op, *args, **kwargs):
        if flag_gems.vendor_name == "kunlunxin" and op is self.torch_op:
            if not hasattr(self, "_native_host_records"):
                self._native_host_records = _load_host_baseline()
            record = self._native_host_records.get(self._native_host_case_id)
            if record is None:
                raise ValueError("No H800 native host baseline for this exact case ID")
            if len(args) != 1 or kwargs:
                raise ValueError(
                    "_lazy_clone host baseline requires its original input ABI"
                )
            inp = args[0]
            config = record["config"]
            actual = {
                "input_shape": list(inp.shape),
                "input_stride": list(inp.stride()),
                "input_storage_offset": inp.storage_offset(),
                "dtype": str(inp.dtype),
                "input_requires_grad": inp.requires_grad,
                "input_is_conj": inp.is_conj(),
                "input_is_neg": inp.is_neg(),
            }
            if actual != {key: config[key] for key in actual}:
                raise ValueError(
                    "_lazy_clone H800 baseline input geometry does not match"
                )
            if inp.untyped_storage().nbytes() != (
                math.prod(config["storage_shape"]) * inp.element_size()
            ):
                raise ValueError(
                    "_lazy_clone H800 baseline storage extent does not match"
                )
            required_device = torch.device(flag_gems.device)
            if required_device.index is None and required_device.type not in (
                "cpu",
                "meta",
            ):
                required_device = torch.device(
                    required_device.type,
                    flag_gems.runtime.torch_device_fn.current_device(),
                )
            if inp.device != required_device or inp.layout != torch.strided:
                raise ValueError(
                    "_lazy_clone candidate input is not on its required backend"
                )
            latency = record["T_host_us"]
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError("Invalid _lazy_clone native host baseline latency")
            return latency / 1000.0
        # Native _lazy_clone creates COW metadata even for accelerator inputs.
        # Match the producer's host timer and object lifetime. No CUDA-event
        # timer, unrelated device synchronization or output materialization.
        measurement = Timer(
            stmt="call()",
            globals={"call": lambda: op(*args, **kwargs)},
            num_threads=1,
            timer=time.perf_counter,
        ).blocked_autorange(min_run_time=0.05)
        return measurement.median * 1000.0

    def set_more_shapes(self):
        return super().set_more_shapes() + LAZY_CLONE_EXTRA_SHAPES


@pytest.mark.underscore_lazy_clone
def test__lazy_clone():
    bench = LazyCloneBenchmark(
        op_name="_lazy_clone",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._lazy_clone,
        gems_op=getattr(flag_gems, "_lazy_clone", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
