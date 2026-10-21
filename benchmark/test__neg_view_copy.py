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
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts

# Select every overload and contract case with the operator marker.
pytestmark = pytest.mark.underscore_neg_view_copy

# _neg_view_copy materializes a fresh copy (negated values) of the input, so
# each case needs input + output (2x one tensor's memory per case). The default
# shape set contains 1G-element core shapes whose fp32 input+output would need
# ~8 GiB and OOM on busy GPUs; cap the shapes while keeping
# performance-relevant sizes (2**26 elements = 256 MiB fp32 per tensor).
MAX_ELEMENTS = 2**26


_H800_ARTIFACT_SHA256 = (
    "a6897c0c65630c1a31573412610157a1765e72025e7aaba60a1d484da45254a3"
)
_H800_SOURCE_COMMIT = "341eaeb6bcf54a24b8b58996cc273173f8525a6c"
_CORE_SHAPES = ((64, 64), (4096, 4096), (64, 512, 512))


@lru_cache(maxsize=1)
def _h800_records():
    # ops-baseline publishes this exact snapshot. Set its local path explicitly;
    # missing or incompatible data must fail rather than time a CPU substitute.
    payload = Path(os.environ["FLAGGEMS_NEG_VIEW_COPY_H800_BASELINE"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != _H800_ARTIFACT_SHA256:
        raise ValueError("Unexpected _neg_view_copy H800 baseline artifact")
    artifact = json.loads(payload)
    environment = artifact["_environment"]
    if (
        environment["hardware"]["gpu"]["name"] != "NVIDIA H800"
        or environment["source"]["git_commit"] != _H800_SOURCE_COMMIT
        or environment["run"]["repo"] != "pytorch"
        or environment["run"]["selected_ops"] != ["_neg_view_copy"]
    ):
        raise ValueError("Baseline provenance does not match the native H800 run")
    return artifact["_neg_view_copy"]["shapes"]


class _H800ReferenceLatency:
    baseline_variant = None

    def _candidate_call(self):
        op, dispatch, overridden = super()._candidate_call()
        if flag_gems.vendor_name == "kunlunxin" and op is self.torch_op:
            raise RuntimeError(
                "Kunlunxin negative-copy benchmark requires an explicit candidate; "
                "the unusable native reference cannot be timed as the candidate"
            )
        return op, dispatch, overridden

    def get_latency(self, op, *args, **kwargs):
        if flag_gems.vendor_name != "kunlunxin" or op is not self.torch_op:
            return super().get_latency(op, *args, **kwargs)
        inp = args[0]
        shape = tuple(inp.shape)
        variant = self.baseline_variant
        if (
            shape not in _CORE_SHAPES
            or not inp.is_contiguous()
            or inp.requires_grad
            or inp.dtype not in consts.FLOAT_DTYPES
            or ("out" in kwargs) != (variant == "out")
        ):
            raise ValueError("Input is outside the collected native H800 core grid")
        if variant == "out":
            out = kwargs["out"]
            if (
                out.shape != inp.shape
                or out.dtype != inp.dtype
                or out.device != inp.device
                or not out.is_contiguous()
            ):
                raise ValueError("Output buffer differs from the H800 input recipe")
        key = variant + "__" + "x".join(map(str, shape))
        record = _h800_records()[key][str(inp.dtype)]
        config = record["config"]
        function = "test__neg_view_copy" + ("_out" if variant == "out" else "")
        case_id = (
            f"benchmark/test__neg_view_copy.py::{function}::core::"
            f"{str(inp.dtype).removeprefix('torch.')}::{_CORE_SHAPES.index(shape)}"
        )
        expected = {
            "operator": "_neg_view_copy",
            "variant": variant,
            "shape": list(shape),
            "dtype": str(inp.dtype),
            "input_recipe": "torch.randn(shape, dtype=dtype, device=device)",
            "input_layout": "contiguous",
            "input_requires_grad": False,
            "out_recipe": "torch.empty_like(input)" if variant == "out" else None,
            "case_id": case_id,
        }
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("Native H800 case identity differs from this benchmark")
        latency_us = record["T_us"]
        if not math.isfinite(latency_us) or latency_us <= 0:
            raise ValueError("Native H800 latency must be positive and finite")
        # Benchmark metrics use milliseconds; ops-baseline T_us uses microseconds.
        return latency_us / 1000.0


class NegViewCopyBenchmark(_H800ReferenceLatency, base.UnaryPointwiseBenchmark):
    # The op is a pure unary pointwise copy, so the public unary pointwise
    # family covers its call semantics (single input tensor -> fresh output).
    baseline_variant = "functional"
    MAX_ELEMENTS = MAX_ELEMENTS

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]


class NegViewCopyOutBenchmark(_H800ReferenceLatency, base.UnaryPointwiseOutBenchmark):
    # The .out overload writes into a caller-supplied buffer; the public unary
    # pointwise-out family builds that buffer and passes it as ``out=``.
    baseline_variant = "out"
    MAX_ELEMENTS = MAX_ELEMENTS

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]


@pytest.mark.underscore_neg_view_copy
def test__neg_view_copy():
    bench = NegViewCopyBenchmark(
        op_name="_neg_view_copy",
        torch_op=torch.ops.aten._neg_view_copy,
        # flag_gems._neg_view_copy is not registered as a direct callable yet;
        # KernelGen's --override _neg_view_copy:<file>:<function> still wins at run
        # time through Benchmark._candidate_call.
        gems_op=getattr(flag_gems, "_neg_view_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.neg_view_copy_out
@pytest.mark.underscore_neg_view_copy
def test__neg_view_copy_out():
    bench = NegViewCopyOutBenchmark(
        op_name="_neg_view_copy",
        torch_op=torch.ops.aten._neg_view_copy.out,
        gems_op=getattr(flag_gems, "_neg_view_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
