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

"""Host-only regressions for scaled_mm collection and capability handling.

Run with --confcutdir=tests/core; no Torch, Triton or device is required.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[2]
ASCEND_ERROR = (
    "'hivm.hir.vbrc' op failed to verify that operand at idx 0 should have "
    "element type f8E4M3FN type or f8E5M2 type or 32-bit float"
)
METAX_ERROR = (
    "CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling cublasLtMatmul "
    "with transpose_mat1 t transpose_mat2 n m 16 n 16 k 16"
)


def _load_modules(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("device operation during collection")

    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module("benchmark", __path__=[str(ROOT / "benchmark")])
    names = (
        "float16",
        "bfloat16",
        "float32",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    module("torch", **{name: f"torch.{name}" for name in names}, arange=unexpected)
    language = module("triton.language", constexpr=object())
    module("triton", language=language, jit=lambda fn: fn)
    gems = module("flag_gems", device="cuda", vendor_name="metax")
    module(
        "flag_gems.runtime", torch_device_fn=SimpleNamespace(current_device=unexpected)
    )
    module("tests.conftest", QUICK_MODE=False)
    module("benchmark.base", Benchmark=object)
    module("benchmark.consts", DEFAULT_METRICS=[])
    config = SimpleNamespace(
        query=False,
        list_cases=False,
        preflight_only=False,
        profile_only=False,
        skip_native=False,
        native_baseline_skip_reason=None,
    )
    conftest = module("benchmark.conftest", Config=config)
    # Exercise the real baseline fixture, including its teardown ordering.
    tree = ast.parse((ROOT / "benchmark/conftest.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in ("configure_native_baseline", "_get_native_baseline_skip_reason")
    ]
    conftest.__dict__.update(pytest=pytest, vendor_name="metax")
    exec(compile(tree, "benchmark/conftest.py", "exec"), conftest.__dict__)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        value = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, value)
        spec.loader.exec_module(value)
        return value

    accuracy = load("tests.test_scaled_mm", "tests/test_scaled_mm.py")
    benchmark = load("benchmark.test_scaled_mm", "benchmark/test_scaled_mm.py")
    return SimpleNamespace(
        accuracy=accuracy,
        benchmark=benchmark,
        config=config,
        conftest=conftest,
        gems=gems,
    )


@pytest.fixture
def modules(monkeypatch):
    return _load_modules(monkeypatch)


def test_import_enumerates_all_cases_without_device_calls(modules):
    accuracy, benchmark = modules.accuracy, modules.benchmark
    assert len(accuracy.CASES) == 80
    assert len(benchmark.CASES) == 240
    assert {p.values[0] for p in accuracy.CASES} == set(accuracy.cases())
    assert {p.values for p in benchmark.CASES} == {
        (case, shape) for case in accuracy.cases() for shape in benchmark.CORE_SHAPES
    }
    assert all(p.id == accuracy.case_id(p.values[0]) for p in accuracy.CASES)
    assert all(
        p.id == f"{accuracy.case_id(p.values[0])}-{p.values[1]}"
        for p in benchmark.CASES
    )


@pytest.mark.parametrize("dtype", ["float8_e4m3fnuz", "float8_e5m2fnuz"])
@pytest.mark.parametrize("stage", ["input_conversion", "output_conversion"])
def test_ascend_known_fnuz_conversion_is_unsupported(modules, dtype, stage):
    modules.gems.vendor_name = "ascend"
    error = type("MLIRCompilationError", (RuntimeError,), {})(ASCEND_ERROR)
    reason = modules.accuracy._unsupported_reason(
        error, dtype=f"torch.{dtype}", stage=stage
    )
    assert dtype in reason and stage in reason


@pytest.mark.parametrize(
    "vendor,dtype,stage,error_type",
    [
        ("metax", "float8_e4m3fnuz", "input_conversion", "MLIRCompilationError"),
        ("ascend", "float8_e4m3fn", "input_conversion", "MLIRCompilationError"),
        ("ascend", "float8_e5m2", "input_conversion", "MLIRCompilationError"),
        ("ascend", "float8_e4m3fnuz", "dot", "MLIRCompilationError"),
        ("ascend", "float8_e4m3fnuz", "allocation", "MLIRCompilationError"),
        ("ascend", "float8_e4m3fnuz", "input_conversion", "RuntimeError"),
    ],
)
def test_unrelated_compiler_errors_are_not_skipped(
    modules, vendor, dtype, stage, error_type
):
    modules.gems.vendor_name = vendor
    error = type(error_type, (RuntimeError,), {})(ASCEND_ERROR)
    with pytest.raises(RuntimeError) as caught:
        modules.accuracy._unsupported_reason(error, dtype=f"torch.{dtype}", stage=stage)
    assert caught.value is error


def test_metax_known_native_rowwise_rejection(modules):
    reason = modules.accuracy._unsupported_reason(
        RuntimeError(METAX_ERROR),
        dtype="torch.float8_e4m3fn",
        stage="native",
        scaling="rowwise",
    )
    assert "native FP8 rowwise" in reason


@pytest.mark.parametrize(
    "vendor,dtype,stage,scaling,message",
    [
        ("nvidia", "float8_e4m3fn", "native", "rowwise", METAX_ERROR),
        ("metax", "float32", "native", "rowwise", METAX_ERROR),
        ("metax", "float8_e4m3fn", "dot", "rowwise", METAX_ERROR),
        ("metax", "float8_e4m3fn", "native", "scalar", METAX_ERROR),
        ("metax", "float8_e4m3fn", "native", "rowwise", "CUBLAS_STATUS_INVALID_VALUE"),
    ],
)
def test_other_native_errors_are_not_skipped(
    modules, vendor, dtype, stage, scaling, message
):
    modules.gems.vendor_name = vendor
    error = RuntimeError(message)
    with pytest.raises(RuntimeError) as caught:
        modules.accuracy._unsupported_reason(
            error, dtype=f"torch.{dtype}", stage=stage, scaling=scaling
        )
    assert caught.value is error


@pytest.mark.parametrize(
    "message",
    [
        "out of memory",
        "device-side assert",
        "illegal memory access",
        "No space left on device",
        "unspecified launch failure",
        "unknown compiler regression",
    ],
)
def test_resource_and_unknown_errors_propagate(modules, message):
    with pytest.raises(RuntimeError, match=message):
        modules.accuracy._unsupported_reason(RuntimeError(message))


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


def _pytest_environment(pytester, monkeypatch):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest(f"""
import importlib.util
import pytest
spec = importlib.util.spec_from_file_location("regression_helpers", {str(Path(__file__))!r})
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
patch = pytest.MonkeyPatch()
modules = helpers._load_modules(patch)
configure_native_baseline = modules.conftest.configure_native_baseline

def pytest_unconfigure(config):
    patch.undo()
""")
    pytester.makeini("""[pytest]
markers =
    scaled_mm
    scaled_mm_out
    unrelated
    skip_native(vendors, reason)
""")


def test_unselected_tests_do_not_probe(pytester, monkeypatch):
    _pytest_environment(pytester, monkeypatch)
    pytester.makepyfile("""
import pytest
from conftest import modules
# Import and collect the real tests, with all device calls prohibited.
for module in (modules.accuracy, modules.benchmark):
    globals().update({k: v for k, v in vars(module).items() if k.startswith("test_")})
@pytest.mark.unrelated
def test_unrelated():
    pass
""")
    result = pytester.runpytest_subprocess("-q", "-m", "unrelated")
    result.assert_outcomes(passed=1, deselected=2242)


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
