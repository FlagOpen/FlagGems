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

"""Check that input capability skips preserve the other benchmark cases."""

import importlib
import json
from collections import Counter
from functools import partial

import pytest
import torch

import flag_gems


@pytest.fixture
def benchmark_config(monkeypatch):
    from benchmark import base, conftest, consts

    config = conftest.BenchConfig()
    config.bench_level = consts.BenchLevel.CORE
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(conftest, "Config", config)
    return base


def _plans(bench):
    return Counter(
        json.dumps(
            (plan.shape, plan.params, plan.builder_args), sort_keys=True, default=str
        )
        for shape in bench.shapes
        for plan in bench.case_fn(shape, torch.float32)
    )


@pytest.mark.parametrize(
    "operator,class_name,group,shapes",
    [
        (
            "_validate_sparse_csr_tensor_args",
            "ValidateSparseCsrTensorArgsBenchmark",
            "empty_values",
            [[4, 4], [0, 3], [3, 0], [0, 3, 4]],
        ),
        (
            "view_as_complex_copy",
            "ViewAsComplexCopyBenchmark",
            "empty_input",
            [[4, 2], [0, 2], [3, 0, 2]],
        ),
    ],
)
def test_empty_partition_preserves_every_plan(
    benchmark_config, tmp_path, operator, class_name, group, shapes
):
    module = importlib.import_module(f"benchmark.test_{operator}")
    shape_file = tmp_path / "shapes.json"
    shape_file.write_text(json.dumps({operator: {"shapes": shapes}}))

    def make(**kwargs):
        bench = getattr(module, class_name)(
            op_name=operator,
            torch_op=lambda *args: None,
            case_fn=module._case_fn,
            build_inputs_fn=module._build_inputs_fn,
            dtypes=[torch.float32],
            **kwargs,
        )
        bench.set_shapes(str(shape_file))
        return _plans(bench)

    nonempty, empty = (make(**{group: value}) for value in (False, True))
    assert nonempty and empty
    assert not (nonempty & empty)
    assert nonempty + empty == make()


def test_layout_partition_preserves_memory_format_queries(benchmark_config, tmp_path):
    from benchmark import test_is_contiguous as module

    shape_file = tmp_path / "shapes.json"
    shape_file.write_text(
        json.dumps({"is_contiguous": {"shapes": [[2, 3, 4, 5], [2, 3, 4, 5, 6]]}})
    )

    def make(layout=None):
        bench = module.IsContiguousBenchmark(
            op_name="is_contiguous",
            torch_op=lambda *args: None,
            case_fn=partial(module._case_fn, input_layout=layout),
            build_inputs_fn=module._build_inputs_fn,
            dtypes=[torch.float32],
            input_layout=layout,
        )
        bench.set_shapes(str(shape_file))
        return _plans(bench)

    combined = Counter()
    for layout in (
        "plain",
        "transposed",
        "column_step",
        "channels_last",
        "channels_last_3d",
    ):
        plans = make(layout)
        assert not (combined & plans)
        combined += plans
    assert combined == make()
    # Querying channels-last on a plain allocation remains a supported case.
    plain = [json.loads(plan)[1] for plan in make("plain")]
    assert {p["memory_format"] for p in plain} == {
        "default",
        "contiguous",
        "preserve",
        "channels_last",
        "channels_last_3d",
    }


@pytest.mark.parametrize("vendor", ["ascend", "nvidia"])
@pytest.mark.parametrize(
    "operator,arguments,unsupported",
    [
        ("_validate_sparse_csr_tensor_args", (torch.float8_e4m3fn, False), True),
        ("_validate_sparse_csr_tensor_args", (torch.float8_e4m3fn, True), False),
        ("is_coalesced", (torch.float8_e4m3fn,), True),
        ("is_coalesced", (torch.float8_e5m2,), True),
        ("is_contiguous", ("channels_last",), True),
        ("is_contiguous", ("plain",), False),
        ("view_as_complex_copy", (torch.float16, False), True),
        ("view_as_complex_copy", (torch.float16, True), False),
    ],
)
def test_capability_skips_are_vendor_and_configuration_scoped(
    benchmark_config, monkeypatch, vendor, operator, arguments, unsupported
):
    module = importlib.import_module(f"benchmark.test_{operator}")
    captured = []
    monkeypatch.setattr(flag_gems, "vendor_name", vendor)
    monkeypatch.setattr(
        benchmark_config.Benchmark, "run", lambda self: captured.append(self)
    )
    name = (
        "test_validate_sparse_csr_tensor_args"
        if operator.startswith("_")
        else f"test_{operator}"
    )
    entry = getattr(module, name)
    if vendor == "ascend" and unsupported:
        with pytest.raises(pytest.skip.Exception):
            entry(*arguments)
        assert not captured
    else:
        entry(*arguments)
        assert len(captured) == 1
