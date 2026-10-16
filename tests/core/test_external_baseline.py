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

"""Synthetic format/contract checks; these are not collected NVIDIA timings."""

import ast
import copy
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.baseline import BaselineData

DTYPE = "torch.float32"
TENSOR = {"shape": [4, 8], "dtype": DTYPE, "stride": [8, 1], "requires_grad": False}
CALL = {"args": [TENSOR, 1], "kwargs": {"keepdim": False}}


def baseline(record=None, key="case0", op="reduce"):
    if record is None:
        record = {"latency_ms": 4.0, "config": {"inputs": copy.deepcopy(CALL)}}
    return BaselineData({op: {"shapes": {key: {DTYPE: record}}}}, "synthetic.json")


def test_nested_metadata_matches_even_when_shape_key_is_a_descriptor():
    call = {"args": [[TENSOR, TENSOR]], "kwargs": {"dim": 0}}
    data = baseline({"latency_ms": 4, "config": {"inputs": copy.deepcopy(call)}})
    assert data.lookup("reduce", DTYPE, call)[1:] == ("case0", None)
    call["kwargs"]["dim"] = 1
    assert data.lookup("reduce", DTYPE, call)[0] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", [8, 4]),
        ("dtype", "torch.int64"),
        ("stride", [1, 4]),
        ("requires_grad", True),
    ],
)
def test_tensor_identity_mismatch_does_not_use_first_shape_shortcut(field, value):
    call = copy.deepcopy(CALL)
    call["args"][0][field] = value
    assert baseline(key="[4, 8]").lookup("reduce", DTYPE, call)[0] is None


@pytest.mark.parametrize("value", [0, True, 2, 1.0])
def test_scalar_type_and_value_mismatches_fall_back(value):
    call = copy.deepcopy(CALL)
    call["args"][1] = value
    assert baseline().lookup("reduce", DTYPE, call)[0] is None


def test_ambiguity_does_not_select_first_record():
    data = baseline()
    shapes = data.data["reduce"]["shapes"]
    shapes["case1"] = copy.deepcopy(shapes["case0"])
    assert data.lookup("reduce", DTYPE, CALL)[2] == "ambiguous_match"


def test_named_inputs_require_binding_and_compare_scalars():
    record = {
        "latency_ms": 1,
        "config": {
            "inputs": {
                "x": TENSOR,
                "dim": {"scalar": 1},
            }
        },
    }
    data = baseline(record)
    assert data.lookup("reduce", DTYPE, CALL)[0] is None
    assert data.lookup("reduce", DTYPE, CALL, {"x": TENSOR, "dim": 1})[0]
    assert data.lookup("reduce", DTYPE, CALL, {"x": TENSOR, "dim": 0})[0] is None


def test_legacy_shape_and_microsecond_units_and_private_operator_names():
    data = baseline({"T_us": 500}, key="[4,8]", op="_reduce")
    assert data.lookup("_reduce", DTYPE, CALL)[0]["latency_ms"] == 0.5
    assert data.lookup("_reduce", "torch.float16", CALL)[0] is None
    assert data.lookup("missing", DTYPE, CALL)[2] == "operator_missing"


@pytest.mark.parametrize(
    "latency", [None, 0, -1, True, "4", float("nan"), float("inf")]
)
def test_invalid_latency_is_a_miss(latency):
    data = baseline({"latency_ms": latency}, key="[4, 8]")
    assert data.lookup("reduce", DTYPE, CALL)[2] == "invalid_latency"


@pytest.mark.parametrize("payload", ["[]", "null", "{broken"])
def test_invalid_file_warns_and_disables_comparison(tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_text(payload)
    with pytest.warns(UserWarning, match="using native baseline"):
        assert BaselineData.load(path) is None
    assert BaselineData.load(None) is None


def test_missing_file_warns_and_valid_metadata_is_preserved(tmp_path):
    with pytest.warns(UserWarning):
        assert BaselineData.load(tmp_path / "missing") is None
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"_environment": {}, "_reduce": {"shapes": {}}}))
    assert "_reduce" in BaselineData.load(path).data


def with_scaling(data):
    data.data["_scaling_factors"] = {
        "chips": [
            {
                "vendor": "nvidia",
                "chip": "H20",
                "measured": {"bandwidth_gbps": 0.5, "vector": {"fp32": 0.25}},
                "nominal": {"tensor": {"fp32": 0.125}},
            },
            {"vendor": "nvidia", "chip": "H800", "measured": {"bandwidth_gbps": 1}},
        ]
    }
    return data


@pytest.mark.parametrize(
    "unit,expected", [("mem", 0.5), ("cuda", 0.25), ("tensor", 0.125)]
)
def test_scaling_selects_chip_and_resource_and_prefers_measured(unit, expected):
    data = with_scaling(baseline())
    assert data.scaling_factor("nvidia", "NVIDIA H20-3e", unit, DTYPE) == expected
    assert data.scaling_factor("nvidia", "NVIDIA H800", "mem", DTYPE) == 1
    assert data.scaling_factor("nvidia", "NVIDIA H100", unit, DTYPE) is None
    assert data.scaling_factor("nvidia", "", unit, DTYPE) is None
    assert data.scaling_factor("ascend", "NVIDIA H20", unit, DTYPE) is None


@pytest.mark.parametrize("factor", [-1, 0, True, float("inf"), "0.5"])
def test_invalid_scaling_falls_back_to_raw_ratio(factor):
    data = with_scaling(baseline())
    data.data["_scaling_factors"]["chips"][0]["measured"]["bandwidth_gbps"] = factor
    assert data.scaling_factor("nvidia", "NVIDIA H20", "mem", DTYPE) is None


def test_external_ratio_preserves_native_measurements_and_records_provenance():
    data = with_scaling(
        baseline(
            {"latency_ms": 4, "bottle_neck_unit": "mem", "config": {"inputs": CALL}}
        )
    )
    metric = SimpleNamespace(latency=2.0, latency_base=6.0, speedup=3.0)
    data.apply(metric, "reduce", DTYPE, CALL, None, "nvidia", "NVIDIA H20")
    assert metric.compared_speedup == 4.0
    assert metric.base_raw_speedup == 2.0
    assert metric.base_scaling_factor == 0.5
    assert metric.base_data_source == "synthetic.json"
    assert metric.latency_base == 6.0 and metric.speedup == 3.0
    assert metric.base_miss_reason is None


def test_skip_native_can_use_external_data_and_a_miss_keeps_na():
    metric = SimpleNamespace(latency=2.0, latency_base=None, speedup=None)
    data = baseline()
    data.apply(metric, "reduce", DTYPE, CALL, None, "unknown", "")
    assert metric.compared_speedup == 2.0 and metric.speedup is None
    missing = SimpleNamespace(latency=2.0, speedup=None)
    data.apply(missing, "missing", DTYPE, CALL, None, "unknown", "")
    assert missing.speedup is None and not hasattr(missing, "compared_speedup")


def scheduler_function(name, namespace):
    """Exercise the real scheduler function without importing device discovery."""
    source = Path(__file__).resolve().parents[2] / "tools" / "run_tests.py"
    tree = ast.parse(source.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("skip_native", [False, True])
def test_real_summary_prefers_external_per_case_and_preserves_fallback(
    tmp_path, skip_native
):
    item = {
        "dtype": DTYPE,
        "result": [
            {
                "shape_detail": [4],
                "latency_base": 6,
                "latency": 2,
                "speedup": 3,
                "compared_speedup": 4,
                "base_latency_ms": 8,
            },
            {
                "shape_detail": [8],
                "latency_base": 2,
                "latency": 1,
                "speedup": 2,
                "base_miss_reason": "shape_or_dtype_missing",
            },
        ],
    }
    if skip_native:
        item["native_baseline_skip_reason"] = "native unavailable"
        item["result"][1]["speedup"] = None
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"reduce": {"result": "passed", "details": [item]}}))
    parse = scheduler_function(
        "parse_perf_data", {"json": json, "consts": SimpleNamespace(DTYPE_MAP={})}
    )
    result = parse("reduce", path)["data"][DTYPE]
    assert result["speedup"] == (4 if skip_native else 3)
    assert result["details"]["[4]"]["native_speedup"] == 3
    assert result["details"]["[8]"]["base_miss_reason"] == "shape_or_dtype_missing"


def test_scheduler_passes_absolute_baseline_path_as_one_shell_argument(tmp_path):
    op_dir = tmp_path / "reduce"
    op_dir.mkdir()
    commands = []
    fn = scheduler_function(
        "run_benchmark_q",
        {
            "ROOT": tmp_path,
            "CFG": SimpleNamespace(output_dir=tmp_path, base_data="/tmp/a b.json"),
            "get_env": lambda _: {},
            "ensure_dir": lambda _: None,
            "op_marker": lambda op: op,
            "time": SimpleNamespace(time=lambda: 1),
            "run_cmd": lambda op, cmd, **kw: commands.append(cmd) or 0,
            "shlex": shlex,
            "TIMEOUT": -100,
        },
    )
    fn(0, "reduce")
    assert "--base-data=/tmp/a b.json" in shlex.split(commands[0])


@pytest.mark.parametrize("config", [[], "invalid", {"inputs": []}, {"inputs": None}])
def test_malformed_metadata_does_not_fall_through_to_legacy_shape(config):
    data = baseline({"latency_ms": 4, "config": config}, key="[4, 8]")
    assert data.lookup("reduce", DTYPE, CALL)[0] is None


def test_malformed_resource_and_oversized_latency_do_not_break_benchmarks():
    data = with_scaling(baseline())
    assert data.scaling_factor("nvidia", "NVIDIA H20", [], DTYPE) is None
    data = baseline({"latency_ms": 10**1000}, key="[4, 8]")
    assert data.lookup("reduce", DTYPE, CALL)[2] == "invalid_latency"
