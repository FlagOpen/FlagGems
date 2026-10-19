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

from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from tests.test_scaled_mm import (
    METAX_ERROR,
    _device_key,
    _load_modules,
    _pytest_environment,
    _unsupported_reason,
    case_id,
    cases,
    comparison_tolerance,
    golden,
    make_inputs,
    require_case,
)

from . import base, conftest, consts


@lru_cache(None)
def _native_reason(case, shape, use_out, device_key, reference):
    if not hasattr(torch, "_scaled_mm"):
        return "torch._scaled_mm is unavailable"
    args, kwargs = make_inputs(case, shape)
    try:
        if use_out:
            out = torch.empty(
                (shape[0], shape[1]), dtype=case[2] or case[0], device=flag_gems.device
            )
            actual = torch.ops.aten._scaled_mm.out(*args, **kwargs, out=out)
        else:
            actual = torch._scaled_mm(*args, **kwargs)
        torch_device_fn.synchronize()
        tol = comparison_tolerance(actual.dtype)
        torch.testing.assert_close(
            actual.cpu().float(), reference(args, kwargs).float(), rtol=tol, atol=tol
        )
    except AssertionError as exc:
        return f"Native probe disagrees with FP32 golden: {str(exc)[:300]}"
    except Exception as exc:
        return _unsupported_reason(exc, dtype=case[0], stage="native", scaling=case[3])
    return None


def native_reason(case, shape, use_out=False, *, reference):
    return _native_reason(case, shape, use_out, _device_key(), reference)


CORE_SHAPES = [(16, 16, 16), (128, 128, 128), (512, 512, 512)]


# Do not compile kernels or call native operators while collecting benchmarks.
CASES = [
    pytest.param(case, shape, id=f"{case_id(case)}-{shape}")
    for case in cases()
    for shape in CORE_SHAPES
]


def _prepare_benchmark(case, shape, use_out, request, monkeypatch):
    config = conftest.Config
    if config.query or config.list_cases:
        return
    require_case(case)
    if config.preflight_only or config.profile_only or config.skip_native:
        return
    reason = native_reason(case, shape, use_out, reference=golden)
    if reason:
        request.node.add_marker(
            pytest.mark.skip_native(vendors=flag_gems.vendor_name, reason=reason)
        )
        # The autouse baseline fixture has already run; update its per-case
        # configuration as well as the reporting marker, then restore it.
        monkeypatch.setattr(config, "skip_native", True)
        monkeypatch.setattr(config, "native_baseline_skip_reason", reason)


class ScaledMMBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, case, shape, use_out):
        self.case, self.shape, self.use_out = case, shape, use_out
        op_name = "scaled_mm_out" if use_out else "scaled_mm"
        op = torch.ops.aten._scaled_mm.out if use_out else torch._scaled_mm
        super().__init__(op_name, op, dtypes=[case[0]])
        self.set_gems(flag_gems.scaled_mm_out if use_out else flag_gems.scaled_mm)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [self.shape]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, dtype):
        args, kwargs = make_inputs(self.case, self.shape)
        if self.use_out:
            kwargs["out"] = torch.empty(
                (self.shape[0], self.shape[1]),
                dtype=self.case[2] or dtype,
                device=flag_gems.device,
            )
        yield (*args, kwargs)

    def record_shapes(self, *args, **kwargs):
        return {
            "fp8_case": case_id(self.case),
            "M_N_K": self.shape,
            "arguments": super().record_shapes(*args, **kwargs),
        }

    def get_tflops(self, op, *args, **kwargs):
        return 2 * args[0].shape[0] * args[0].shape[1] * args[1].shape[1]


@pytest.mark.scaled_mm
@pytest.mark.parametrize("case,shape", CASES)
def test_scaled_mm_benchmark(case, shape, request, monkeypatch):
    _prepare_benchmark(case, shape, False, request, monkeypatch)
    ScaledMMBenchmark(case, shape, False).run()


@pytest.mark.scaled_mm_out
@pytest.mark.parametrize("case,shape", CASES)
def test_scaled_mm_out_benchmark(case, shape, request, monkeypatch):
    _prepare_benchmark(case, shape, True, request, monkeypatch)
    ScaledMMBenchmark(case, shape, True).run()


# Host-only regression checks: no capability probe may run during collection.
pytest_plugins = ["pytester"]


@pytest.fixture
def modules(monkeypatch):
    return _load_modules(monkeypatch)


@pytest.mark.parametrize("out", [False, True])
def test_native_probe_keeps_case_context(modules, monkeypatch, out):
    module = modules.benchmark
    case = (
        "torch.float8_e4m3fn",
        "torch.float8_e4m3fn",
        "torch.bfloat16",
        "rowwise",
        True,
    )

    def rejected(*args, **kwargs):
        raise RuntimeError(METAX_ERROR)

    monkeypatch.setattr(module, "make_inputs", lambda *args: ((), {}))
    monkeypatch.setattr(module.torch, "_scaled_mm", rejected, raising=False)
    monkeypatch.setattr(
        module.torch, "empty", lambda *args, **kwargs: object(), raising=False
    )
    monkeypatch.setattr(
        module.torch,
        "ops",
        SimpleNamespace(aten=SimpleNamespace(_scaled_mm=SimpleNamespace(out=rejected))),
        raising=False,
    )
    assert "native FP8 rowwise" in module._native_reason(
        case, (16, 16, 16), out, ("cuda", 0), object()
    )


def test_runtime_skips_errors_and_baseline_state(pytester, monkeypatch):
    _pytest_environment(pytester, monkeypatch)
    pytester.makepyfile("""
import pytest
from conftest import modules

def check(value):
    if value == "unsupported":
        return "known capability limitation"
    if value == "broken":
        raise RuntimeError("unknown probe failure")
    return None
modules.accuracy.case_reason = check
modules.benchmark.native_reason = lambda case, shape, use_out, **kw: (
    "native unavailable" if case == "no-native" else None
)
@pytest.mark.parametrize("case", ["supported", "unsupported", "broken"])
def test_accuracy(case):
    modules.accuracy.require_case(case)
    assert case == "supported"
@pytest.mark.parametrize("case", ["no-native", "supported", "unsupported", "broken", "supported"])
@pytest.mark.parametrize("shape", [(16, 16, 16)])
def test_benchmark(case, shape, request, monkeypatch):
    modules.benchmark._prepare_benchmark(case, shape, False, request, monkeypatch)
    assert modules.config.skip_native == (case == "no-native")
    assert modules.config.native_baseline_skip_reason == ("native unavailable" if case == "no-native" else None)
def test_final_reset():
    assert modules.config.skip_native is False
    assert modules.config.native_baseline_skip_reason is None
""")
    result = pytester.runpytest_subprocess("-q")
    result.assert_outcomes(passed=5, skipped=2, failed=2)
    assert "unknown probe failure" in result.stdout.str()


@pytest.mark.parametrize(
    "mode", ["query", "list_cases", "preflight_only", "profile_only", "skip_native"]
)
def test_metadata_and_candidate_modes_do_not_probe_native(modules, monkeypatch, mode):
    monkeypatch.setattr(modules.config, mode, True)
    calls = []
    monkeypatch.setattr(
        modules.benchmark, "require_case", lambda case: calls.append(case)
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("unexpected native probe")

    monkeypatch.setattr(modules.benchmark, "native_reason", unexpected)
    modules.benchmark._prepare_benchmark("case", (16, 16, 16), False, None, monkeypatch)
    assert calls == ([] if mode in ("query", "list_cases") else ["case"])


@pytest.mark.parametrize("use_out", [False, True])
def test_native_skip_still_runs_the_selected_candidate(modules, monkeypatch, use_out):
    module = modules.benchmark
    calls = []
    markers = []
    monkeypatch.setattr(module, "require_case", lambda case: None)

    def native(case, shape, out, **kwargs):
        assert out is use_out
        return "native scaling unsupported"

    class Candidate:
        def __init__(self, case, shape, out):
            assert out is use_out

        def run(self):
            assert modules.config.skip_native is True
            assert (
                modules.config.native_baseline_skip_reason
                == "native scaling unsupported"
            )
            calls.append("candidate ran")

    monkeypatch.setattr(module, "native_reason", native)
    monkeypatch.setattr(module, "ScaledMMBenchmark", Candidate)
    request = SimpleNamespace(node=SimpleNamespace(add_marker=markers.append))
    test = (
        module.test_scaled_mm_out_benchmark
        if use_out
        else module.test_scaled_mm_benchmark
    )
    test("case", (16, 16, 16), request, monkeypatch)
    assert calls == ["candidate ran"]
    assert markers[0].name == "skip_native"
