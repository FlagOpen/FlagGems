"""Exercise the production case loop without importing a device runtime."""

import ast
import gc
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "reference_contract", ROOT / "benchmark/reference.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class OutOfMemoryError(RuntimeError):
    pass


def make_runner(errors, *, sync_error=None, fail_build=False):
    """Compile the real method, replacing only its runtime dependencies."""
    tree = ast.parse((ROOT / "benchmark/base.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Benchmark"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_reference_cases"
    )
    events = []
    config = SimpleNamespace(
        reference_records=[],
        available_case_ids=set(),
        executed_case_ids=set(),
        current_nodeid="test",
        skip_native=False,
    )

    def synchronize():
        events.append("sync")
        if sync_error:
            raise sync_error

    namespace = {
        "Config": config,
        "reference_failure": reference.reference_failure,
        "pytest": pytest,
        "math": math,
        "gc": gc,
        "torch": SimpleNamespace(OutOfMemoryError=OutOfMemoryError),
        "Optional": __import__("typing").Optional,
        "Collection": __import__("typing").Collection,
        "torch_device_fn": SimpleNamespace(
            synchronize=synchronize, empty_cache=lambda: events.append("empty_cache")
        ),
    }
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(ROOT / "benchmark/base.py"),
            "exec",
        ),
        namespace,
    )
    cases = []
    for index in range(len(errors)):
        data = {
            "case_id": str(index),
            "ordinal": index,
            "dtype": "float32",
            "shape": [8],
            "params": {},
        }
        cases.append(
            SimpleNamespace(case_id=str(index), to_dict=lambda data=data: data.copy())
        )

    def latency(op, index):
        events.append(index)
        error = errors[index]
        if error is not None:
            raise error
        return 1.0

    def build_inputs(case):
        index = int(case.case_id)
        if fail_build and errors[index] is not None:
            raise errors[index]
        return index

    bench = SimpleNamespace(
        op_name="op",
        supports_cases=lambda: True,
        _collect_cases=lambda: cases,
        build_inputs=build_inputs,
        unpack_to_args_kwargs=lambda x: ((x,), {}),
        torch_op=object(),
        get_latency=latency,
    )
    return lambda: namespace["_run_reference_cases"](bench, None), config, events


def test_oom_dtype_and_missing_api_are_all_collected_before_success():
    missing = AttributeError("missing", name="missing", obj=ModuleType("torch"))
    run, config, events = make_runner(
        [
            OutOfMemoryError("allocation failed"),
            RuntimeError("op not implemented for 'BFloat16'"),
            missing,
            None,
        ]
    )
    with pytest.raises(pytest.fail.Exception, match="3 reference cases failed"):
        run()
    records = config.reference_records
    assert [r["status"] for r in records] == ["FAILED", "FAILED", "FAILED", "PASSED"]
    assert [r["failure"]["category"] for r in records[:3]] == [
        "OUT_OF_MEMORY",
        "DTYPE_UNSUPPORTED",
        "API_MISSING",
    ]
    assert [r["failure"]["skip_eligible"] for r in records[:3]] == [True, True, False]
    assert "empty_cache" in events and 3 in events
    assert (
        reference.reference_report(json.loads(json.dumps(records)))["status"]
        == "FAILED"
    )


@pytest.mark.parametrize(
    "error",
    [
        TypeError("bad signature"),
        ValueError("bad shape"),
        IndexError("out of range"),
        AssertionError("dtype mismatch"),
        RuntimeError("unknown failure"),
    ],
)
def test_other_case_errors_continue_but_cannot_be_skipped(error):
    run, config, events = make_runner([error, None])
    with pytest.raises(pytest.fail.Exception):
        run()
    assert config.reference_records[1]["status"] == "PASSED"
    assert config.reference_records[0]["failure"]["skip_eligible"] is False
    assert "empty_cache" not in events


def test_failed_oom_recovery_stops_and_revokes_skip_eligibility():
    run, config, _ = make_runner(
        [OutOfMemoryError("OOM"), None], sync_error=RuntimeError("device lost")
    )
    with pytest.raises(RuntimeError, match="device lost"):
        run()
    assert [r["status"] for r in config.reference_records] == ["FAILED", "NOT_RUN"]
    assert config.reference_records[0]["failure"]["skip_eligible"] is False
    assert config.reference_records[0]["recovery_failure"]["category"] == "DEVICE_ERROR"


@pytest.mark.parametrize(
    "error",
    [
        KeyboardInterrupt(),
        MemoryError("host allocation"),
        RuntimeError("CUDA error: device-side assert triggered"),
    ],
)
def test_unrecoverable_errors_stop_traversal(error):
    run, config, events = make_runner([error, None])
    with pytest.raises(type(error)):
        run()
    assert config.reference_records[1]["status"] == "NOT_RUN"
    assert 1 not in events


def test_case_skip_continues_and_all_skip_never_passes():
    run, config, _ = make_runner([pytest.skip.Exception("explicit source rule"), None])
    assert run() == ["1"]
    assert [r["status"] for r in config.reference_records] == ["SKIP", "PASSED"]
    run, config, _ = make_runner([pytest.skip.Exception("explicit source rule")])
    assert run() == []
    assert reference.reference_report(config.reference_records)["status"] == "ALL_SKIP"


@pytest.mark.parametrize("fail_build", [False, True])
def test_multiple_oom_cases_are_collected_in_one_traversal(fail_build):
    run, config, events = make_runner(
        [OutOfMemoryError("first"), None, OutOfMemoryError("second"), None],
        fail_build=fail_build,
    )
    with pytest.raises(pytest.fail.Exception, match="2 reference cases failed"):
        run()
    assert [r["status"] for r in config.reference_records] == [
        "FAILED", "PASSED", "FAILED", "PASSED"
    ]
    assert events.count("empty_cache") == 2
    assert config.reference_records[0]["stage"] == (
        "build_inputs" if fail_build else "benchmark_reference"
    )


def test_timer_sync_failure_stops_even_if_it_looks_like_oom():
    error = OutOfMemoryError("async failure")
    error._reference_sync_failure = True
    run, config, _ = make_runner([error, None])
    with pytest.raises(OutOfMemoryError):
        run()
    assert config.reference_records[0]["stage"] == "synchronize"
    assert config.reference_records[0]["failure"]["skip_eligible"] is False
    assert config.reference_records[1]["status"] == "NOT_RUN"
