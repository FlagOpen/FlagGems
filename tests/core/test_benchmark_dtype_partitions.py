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

"""Exercise dtype selection through the real benchmark initialization path."""

import json

import pytest
import torch

from benchmark import base, conftest, consts

DTYPES = [torch.float16, torch.float32]


def forbidden(*args, **kwargs):
    raise AssertionError("metadata enumeration must not construct or execute tensors")


@pytest.fixture
def config(monkeypatch, tmp_path):
    config = conftest.BenchConfig()
    config.bench_level = consts.BenchLevel.CORE
    config.list_cases = True
    config.current_nodeid = "benchmark/test_dtype_partition.py::test_partition"
    shapes = tmp_path / "shapes.json"
    shapes.write_text(json.dumps({"dtype_partition": {"shapes": [[2, 3]]}}))
    config.shape_file = str(shapes)
    monkeypatch.setattr(base, "Config", config)
    monkeypatch.setattr(conftest, "Config", config)
    monkeypatch.setattr(base, "update_case_list", lambda record: None)
    return config


def make_benchmark(partition_dtype=None):
    def plans(shape, dtype):
        yield base.BenchmarkCasePlan(shape={"input": shape}, params={}, builder_args=())

    return base.GenericBenchmark(
        "dtype_partition",
        torch_op=forbidden,
        dtypes=DTYPES,
        partition_dtype=partition_dtype,
        case_fn=plans,
        build_inputs_fn=forbidden,
    )


@pytest.mark.parametrize("requested", [None, [torch.float16], [torch.float32], DTYPES])
@pytest.mark.parametrize("partition", DTYPES)
def test_run_filters_valid_requests_without_rejecting_other_partitions(
    config, requested, partition
):
    config.user_desired_dtypes = requested
    bench = make_benchmark(partition)
    if requested is not None and partition not in requested:
        with pytest.raises(pytest.skip.Exception, match="was not requested"):
            bench.run()
    else:
        cases = bench.run().cases
        assert len(cases) == 1
        assert {case.dtype for case in cases} == {partition}
    assert bench.dtypes == DTYPES


@pytest.mark.parametrize("requested", [[torch.int8], [torch.float32, torch.int8]])
@pytest.mark.parametrize("partition", DTYPES)
def test_invalid_request_is_rejected_before_partition_filter(
    config, requested, partition
):
    config.user_desired_dtypes = requested
    with pytest.raises(ValueError, match="torch.int8"):
        make_benchmark(partition).run()


def test_unpartitioned_benchmark_keeps_multiple_requested_dtypes(config):
    config.user_desired_dtypes = DTYPES
    assert {case.dtype for case in make_benchmark().run().cases} == set(DTYPES)


def test_partition_must_belong_to_operator_dtype_set(config):
    with pytest.raises(ValueError, match="partition_dtype"):
        make_benchmark(torch.int8)


@pytest.mark.parametrize(
    "mode,executor",
    [
        ("reference_only", "_run_reference_cases"),
        ("preflight_only", "_run_preflight_cases"),
        ("timing", "_run_cases"),
    ],
)
def test_execution_modes_receive_only_selected_partition(
    config, monkeypatch, mode, executor
):
    config.list_cases = False
    config.user_desired_dtypes = DTYPES
    if mode != "timing":
        setattr(config, mode, True)
    bench = make_benchmark(torch.float32)
    monkeypatch.setattr(bench, executor, lambda selection: list(bench.to_bench_dtypes))
    assert bench.run() == [torch.float32]


@pytest.mark.parametrize("requested", [[torch.float32], [torch.float16, torch.float32]])
def test_adjoint_entry_runs_real_dtype_selection(config, monkeypatch, requested):
    import flag_gems
    from benchmark import test_adjoint as module

    config.user_desired_dtypes = requested
    monkeypatch.setattr(flag_gems, "vendor_name", "nvidia")
    monkeypatch.setattr(module, "_build_inputs_fn", forbidden)
    records = []
    monkeypatch.setattr(base, "update_case_list", records.append)
    for dtype in module.BENCH_DTYPES:
        if dtype in requested:
            module.test_adjoint(dtype)
        else:
            with pytest.raises(pytest.skip.Exception, match="was not requested"):
                module.test_adjoint(dtype)
    assert {case["dtype"] for record in records for case in record["cases"]} == {
        str(dtype) for dtype in requested
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_backward_query_preserves_cross_cast_partition(config, dtype):
    from benchmark import test_to_mkldnn_backward as module

    bench = module.ToMkldnnBackwardBenchmark(
        op_name="to_mkldnn_backward",
        torch_op=forbidden,
        case_fn=module._case_fn,
        build_inputs_fn=forbidden,
        dtypes=module.BENCH_DTYPES,
        partition_dtype=dtype,
        case_group=module.OPAQUE_CROSS_CAST_GROUP,
    )
    # --query initializes shapes without first calling set_dtypes.
    bench.DEFAULT_SHAPE_FILES = config.shape_file
    bench.init_default_config()
    assert bench.shapes
    assert all(
        any(module._case_fn(shape, dtype, module.OPAQUE_CROSS_CAST_GROUP))
        for shape in bench.shapes
    )
