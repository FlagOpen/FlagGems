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

"""Sparse-mask timing, optionally against a recorded NVIDIA baseline.

Set FLAGGEMS_NV_BASELINE to a local ops-baseline JSON file to import the matched
native latency. Candidates still execute on the target; JSON records identify
both devices and label the ratio as a raw cross-chip comparison.
"""

import hashlib
import json
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import pytest
import torch
from torch.autograd.function import once_differentiable

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


def _copy_dense_physical(value, device):
    assert isinstance(value, torch.Tensor) and value.layout == torch.strided
    shape, stride, offset = tuple(value.shape), value.stride(), value.storage_offset()
    source_dtype = value.dtype
    source_conj, source_neg = value.is_conj(), value.is_neg()
    source_storage = value.untyped_storage()
    raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
        source_storage, 0, (source_storage.nbytes(),), (1,)
    )
    # Independent storage on CPU even when the input is already CPU.
    raw = raw.to(device=device).clone()
    result = raw.view(source_dtype).as_strided(shape, stride, offset)
    if source_conj:
        result = result.conj()
    if source_neg:
        result = torch._neg_view(result)
    assert result.device == torch.device(device) and result.dtype == source_dtype
    assert tuple(result.shape) == shape and result.stride() == stride
    assert result.storage_offset() == offset
    assert result.is_conj() == source_conj and result.is_neg() == source_neg
    return result


def _copy_tensor_physical(value, device, *, preserve_requires_grad=False):
    assert isinstance(value, torch.Tensor)
    required_shape, required_dtype = tuple(value.shape), value.dtype
    if value.layout == torch.sparse_coo:
        result = torch.sparse_coo_tensor(
            _copy_dense_physical(value._indices(), device),
            _copy_dense_physical(value._values(), device),
            required_shape,
            dtype=required_dtype,
            device=device,
            is_coalesced=value.is_coalesced(),
        )
        assert result.sparse_dim() == value.sparse_dim()
        assert result.dense_dim() == value.dense_dim()
        assert result.is_coalesced() == value.is_coalesced()
        assert result._nnz() == value._nnz()
    elif value.layout == torch.strided:
        result = _copy_dense_physical(value, device)
    else:
        # Original cases use dense or COO; do not invent a compressed conversion.
        raise AssertionError(
            "CPU bootstrap transport supports original dense/COO layouts only"
        )
    assert result.device == torch.device(device)
    assert result.dtype == required_dtype and tuple(result.shape) == required_shape
    assert result.layout == value.layout
    if preserve_requires_grad and value.requires_grad:
        result.requires_grad_(True)
    return result


def _copy_tensor(value, device, *, preserve_requires_grad=False):
    # Auxiliary transport failure must not satisfy an original negative caller's
    # broad RuntimeError expectation. Actual sparse_mask/autograd calls stay out.
    try:
        return _copy_tensor_physical(
            value, device, preserve_requires_grad=preserve_requires_grad
        )
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError("sparse_mask CPU bootstrap transport failed") from error


def _assert_return_ABI(result, cpu_result, required_device):
    assert isinstance(result, torch.Tensor) and isinstance(cpu_result, torch.Tensor)
    assert result.device == required_device
    assert result.dtype == cpu_result.dtype and result.shape == cpu_result.shape
    assert result.layout == cpu_result.layout
    if result.layout == torch.sparse_coo:
        assert result.is_coalesced() == cpu_result.is_coalesced()
        assert result.sparse_dim() == cpu_result.sparse_dim()
        assert result.dense_dim() == cpu_result.dense_dim()
        assert result._nnz() == cpu_result._nnz()
        for part in (result._indices(), result._values()):
            assert part.device == required_device
        assert result._indices().dtype == cpu_result._indices().dtype
        assert result._values().dtype == cpu_result._values().dtype


class _SparseMaskCPUBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, mask):
        required_device = inp.device
        cpu_inp = _copy_tensor(inp, "cpu")
        cpu_mask = _copy_tensor(mask, "cpu")
        # Native CPU semantics decide dtype promotion, sparse support and errors.
        cpu_result = torch.ops.aten.sparse_mask(cpu_inp, cpu_mask)
        ctx.save_for_backward(cpu_inp, cpu_mask, mask)
        ctx.required_device = required_device
        ctx.input_dtype, ctx.input_shape, ctx.input_layout = (
            inp.dtype,
            inp.shape,
            inp.layout,
        )
        result = _copy_tensor(cpu_result, required_device)
        _assert_return_ABI(result, cpu_result, required_device)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Accessing saved_tensors checks the original mask version, as native does.
        cpu_inp, cpu_mask, _original_mask = ctx.saved_tensors
        assert grad_output.device == ctx.required_device
        cpu_grad = _copy_tensor(grad_output, "cpu")
        # Function.forward executes with grad disabled. Recompute only genuine
        # CPU native under enable_grad to preserve its first-order gradient.
        with torch.enable_grad():
            cpu_leaf = _copy_tensor(cpu_inp, "cpu").requires_grad_(True)
            cpu_result = torch.ops.aten.sparse_mask(cpu_leaf, cpu_mask)
            (cpu_input_grad,) = torch.autograd.grad(
                cpu_result,
                cpu_leaf,
                grad_outputs=cpu_grad,
                create_graph=False,
                retain_graph=False,
            )
        result = _copy_tensor(cpu_input_grad, ctx.required_device)
        assert result.device == ctx.required_device
        assert result.dtype == ctx.input_dtype and result.shape == ctx.input_shape
        assert result.layout == ctx.input_layout
        return result, None


def sparse_mask_cpu_native(inp, mask, *, out=None):
    """NativeCPU bootstrap candidate; original argument tensors are never edited.

    Original sparse_mask tests pass bool COO masks and dense/COO self. The out
    cases have no autograd requirement; no claim about out-autograd is made.
    """
    assert isinstance(inp, torch.Tensor) and isinstance(mask, torch.Tensor)
    required_device = inp.device  # Capture original tensor device before copies.
    if mask.device != required_device:
        raise RuntimeError("self and mask devices must match before CPU transport")
    if out is None:
        return _SparseMaskCPUBridge.apply(inp, mask)
    assert isinstance(out, torch.Tensor)
    if out.device != required_device:
        raise RuntimeError("out and self devices must match before CPU transport")
    cpu_inp = _copy_tensor(inp, "cpu", preserve_requires_grad=True)
    cpu_mask = _copy_tensor(mask, "cpu", preserve_requires_grad=True)
    cpu_out = _copy_tensor(out, "cpu", preserve_requires_grad=True)
    returned = torch.ops.aten.sparse_mask.out(cpu_inp, cpu_mask, out=cpu_out)
    assert returned is cpu_out
    target = _copy_tensor(returned, required_device)
    _assert_return_ABI(target, returned, required_device)
    # Native out may overwrite/reallocate. Promise only original object identity,
    # not stable StorageImpl/data_ptr or readonly out components.
    out.data = target
    _assert_return_ABI(out, returned, required_device)
    return out


_cpu_oracle = sparse_mask_cpu_native

# sparse_mask gathers values from a dense ``self`` at a sparse mask's index
# positions and returns a sparse COO tensor. There is no public Benchmark family
# for sparse gather ops, so the benchmark uses the two-phase GenericBenchmark
# (case_fn + build_inputs_fn): case_fn yields one BenchmarkCasePlan per
# (shape, nnz_ratio) workload and build_inputs_fn materializes the dense self
# and the coalesced sparse mask lazily. The candidate is called directly from
# flag_gems.sparse_mask, with an optional process-local explicit override.
# The perf reference is the aten op itself
# (torch.ops.aten.sparse_mask) and both are called with the same (self, mask)
# signature.
#
# The benchmark work scales with the number of mask nonzeros and the size of the
# dense self, so the shapes pair moderate square layouts with a fixed nnz ratio;
# the ratio drops for the largest shapes to keep the mask/build cost bounded.
_SPARSE_MASK_SHAPES = [
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
]


def _nnz_ratio(shape):
    # 10% nonzeros on the small layouts, 2% on the large ones.
    return 0.1 if math.prod(shape) <= 1024 * 1024 else 0.02


def _case_fn(shape, dtype):
    # yield generates one BenchmarkCasePlan per (shape, dtype) parametrization.
    del dtype
    ratio = _nnz_ratio(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": shape, "mask": shape},
        params={"nnz_ratio": ratio},
        builder_args=(shape, ratio),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, nnz_ratio = plan.builder_args
    self_t = torch.randn(shape, dtype=dtype, device=device)
    if os.environ.get("FLAGGEMS_NV_BASELINE"):
        # Same explicit PR17 sample in every backend/process, without global RNG changes.
        generator = torch.Generator(device="cpu").manual_seed(20261019 + shape[0])
        mask = (
            (torch.rand(shape, generator=generator) > (1.0 - nnz_ratio))
            .to_sparse()
            .to(device)
        )
    else:
        mask = (torch.rand(shape, device=device) > (1.0 - nnz_ratio)).to_sparse()
    return self_t, mask, {}


@dataclass
class _CrossMachineMetrics(base.BenchmarkMetrics):
    baseline_source: dict | None = None


class SparseMaskBenchmark(OperatorBenchmark):
    def _candidate_call(self):
        registry = base.Config.override_registry
        override = registry.get_override(self.op_name) if registry is not None else None
        candidate = override if override is not None else self.gems_op
        if not callable(candidate):
            raise RuntimeError(f"No callable candidate for {self.op_name}")
        return candidate, nullcontext(), override is not None

    # sparse_mask has no meaningful dense shapes in core_shapes.yaml, so
    # benchmark the dedicated square layouts above.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_SPARSE_MASK_SHAPES)

    def _external_input_config(self, args):
        if len(args) != 2:
            raise ValueError(
                "Sparse baseline requires dense self and a boolean COO mask"
            )
        inp, mask = args
        if (
            inp.layout != torch.strided
            or mask.layout != torch.sparse_coo
            or mask.dtype != torch.bool
            or inp.device != mask.device
            or not mask.is_coalesced()
            or mask.sparse_dim() != 2
            or mask.dense_dim() != 0
        ):
            raise ValueError(
                "Sparse baseline requires the original coalesced boolean COO mask"
            )
        # Read actual target indices back before comparing their replay identity.
        indices = mask.indices().detach().cpu().contiguous()
        if not bool(mask.values().detach().cpu().all()):
            raise ValueError("Sparse baseline mask values must all be true")
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
            "benchmark": "benchmark/test_sparse_mask.py",
            "descriptor": list(inp.shape),
            "scope": "core",
            "nnz_ratio": _nnz_ratio(tuple(inp.shape)),
            "mask": {
                "seed": 20261019 + inp.shape[0],
                "generator": "torch CPU default generator",
                "uniform_dtype": "torch.float32",
                "indices_sha256": hashlib.sha256(
                    indices.numpy().astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "nnz": mask._nnz(),
                "index_encoding": "contiguous int64 little-endian COO rows then columns",
                "layout": "torch.sparse_coo",
                "sparse_dim": 2,
                "dense_dim": 0,
                "coalesced": True,
            },
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


@pytest.mark.sparse_mask
def test_sparse_mask():
    bench = SparseMaskBenchmark(
        op_name="sparse_mask",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_mask,
        gems_op=flag_gems.sparse_mask,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
