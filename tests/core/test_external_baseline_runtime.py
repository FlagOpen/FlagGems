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

"""Benchmark integration with real CPU tensors and deterministic test timers."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from benchmark import base, conftest
from benchmark.baseline import BaselineData, describe_call
from benchmark.consts import BenchmarkResult


@pytest.mark.parametrize("skip_native", [False, True])
@pytest.mark.parametrize("matched", [False, True])
def test_measurement_preserves_native_execution_and_skip_contract(
    monkeypatch, skip_native, matched
):
    config = conftest.BenchConfig()
    config.skip_native = skip_native
    inputs = (torch.ones(4, 8),)
    shape = "[4, 8]" if matched else "[8, 4]"
    config.base_data = BaselineData(
        {"example": {"shapes": {shape: {"torch.float32": {"latency_ms": 8.0}}}}},
        "synthetic.json",
    )
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(
        base, "torch_device_fn", SimpleNamespace(get_device_name=lambda: "NVIDIA H20")
    )
    reference = lambda x: x
    candidate = lambda x: x
    bench = base.Benchmark("example", torch_op=reference, gems_op=candidate)
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    monkeypatch.setattr(
        bench, "_candidate_call", lambda: (candidate, nullcontext(), False)
    )
    calls = []

    def latency(op, *args, **kwargs):
        calls.append(op)
        return 6.0 if op is reference else 2.0

    monkeypatch.setattr(bench, "get_latency", latency)
    metric = bench._measure_input(inputs, dtype=torch.float32)
    assert calls == ([candidate] if skip_native else [reference, candidate])
    assert metric.latency == 2.0
    assert metric.latency_base == (None if skip_native else 6.0)
    assert metric.speedup == (None if skip_native else pytest.approx(3.0))
    assert metric.compared_speedup == (4.0 if matched else None)
    report = BenchmarkResult("example", "torch.float32", "kernel", "core", [metric])
    assert ("vs Base" in str(report)) == matched


def test_named_binding_handles_default_parameters(monkeypatch):
    config = conftest.BenchConfig()
    x = torch.ones(4, 8)
    config.base_data = BaselineData(
        {
            "example": {
                "shapes": {
                    "semantic-key": {
                        "torch.float32": {
                            "latency_ms": 4,
                            "config": {
                                "inputs": {
                                    "x": {"shape": [4, 8], "dtype": "torch.float32"},
                                    "dim": {"scalar": 1},
                                }
                            },
                        }
                    }
                }
            }
        },
        "synthetic.json",
    )
    monkeypatch.setattr(base, "Config", config)

    def native(x, dim=1):
        return x

    bench = base.Benchmark("example", torch_op=native, gems_op=native)
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    monkeypatch.setattr(
        bench, "_candidate_call", lambda: (native, nullcontext(), False)
    )
    monkeypatch.setattr(bench, "get_latency", lambda *a, **kw: 2)
    metric = bench._measure_input((x,), dtype=torch.float32)
    assert metric.compared_speedup == 2


def test_no_baseline_does_not_describe_inputs_or_query_device(monkeypatch):
    config = conftest.BenchConfig()
    monkeypatch.setattr(base, "Config", config)

    def forbidden(*a, **kw):
        raise AssertionError("optional baseline path must stay disabled")

    monkeypatch.setattr(base, "describe_call", forbidden)
    monkeypatch.setattr(
        base, "torch_device_fn", SimpleNamespace(get_device_name=forbidden)
    )
    native = lambda x: x
    bench = base.Benchmark("example", torch_op=native, gems_op=native)
    bench.to_bench_metrics = ["latency_base", "latency", "speedup"]
    monkeypatch.setattr(
        bench, "_candidate_call", lambda: (native, nullcontext(), False)
    )
    monkeypatch.setattr(bench, "get_latency", lambda *a, **kw: 2)
    metric = bench._measure_input((torch.ones(4),), dtype=torch.float32)
    assert metric.speedup == pytest.approx(1) and metric.compared_speedup is None


def test_real_tensor_descriptors_keep_layout_nested_roles_and_mixed_dtypes():
    x = torch.ones(4, 8).t()
    call = describe_call(([x, torch.ones(2, dtype=torch.int64)],), {"dim": 0})
    assert call["args"][0][0]["stride"] == [1, 8]
    assert call["args"][0][1]["dtype"] == "torch.int64"
    assert call["kwargs"] == {"dim": 0}


def test_pytest_option_loads_baseline_and_can_reset_to_disabled(tmp_path, monkeypatch):
    import json

    from _pytest.config.argparsing import Parser

    path = tmp_path / "baseline with spaces.json"
    path.write_text(json.dumps({"example": {"shapes": {}}}))
    parser = Parser(_ispytest=True)
    conftest.pytest_addoption(parser)
    monkeypatch.setattr(conftest, "Config", conftest.Config)
    monkeypatch.setattr(conftest, "REPORT_FILE", conftest.REPORT_FILE)
    monkeypatch.setattr(conftest, "apply_overrides_from_args", lambda _: None)
    for argv in [[f"--base-data={path}"], []]:
        options = parser.parse(argv)
        config = SimpleNamespace(
            option=options,
            addinivalue_line=lambda *a: None,
            getini=lambda _: [],
            getoption=lambda key: getattr(options, key.lstrip("-").replace("-", "_")),
            hook=SimpleNamespace(pytest_flaggems_profile_scope=lambda **kw: None),
        )
        conftest.pytest_configure(config)
        if argv:
            assert conftest.Config.base_data.source == str(path)
        else:
            assert conftest.Config.base_data is None
