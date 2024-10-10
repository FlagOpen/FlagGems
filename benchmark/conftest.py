import json
import logging

import pytest

from .attri_util import (
    ALL_AVAILABLE_METRICS,
    BOOL_DTYPES,
    DEFAULT_ITER_COUNT,
    DEFAULT_WARMUP_COUNT,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
    OperationAttribute,
    get_recommended_shapes,
)


class BenchConfig:
    def __init__(self):
        self.cpu_mode = False
        self.bench_level = BenchLevel.COMPREHENSIVE
        self.warm_up = DEFAULT_WARMUP_COUNT
        self.repetition = DEFAULT_ITER_COUNT
        self.record_log = False
        self.user_desired_dtypes = None
        self.user_desired_metrics = None


Config = BenchConfig()


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="cuda",
        required=False,
        choices=["cuda", "cpu"],
        help=(
            "Specify how to measure latency, "
            "'cpu' for CPU-side measurement or 'cuda' for GPU-side measurement."
        ),
    )

    parser.addoption(
        "--level",
        action="store",
        default="comprehensive",
        required=False,
        choices=[level.value for level in BenchLevel],
        help="Specify the benchmark level: comprehensive, or core.",
    )

    parser.addoption(
        "--warmup",
        default=DEFAULT_WARMUP_COUNT,
        help="Number of warmup runs before benchmark run.",
    )

    parser.addoption(
        "--iter",
        default=DEFAULT_ITER_COUNT,
        help="Number of reps for each benchmark run.",
    )

    parser.addoption(
        "--query", action="store_true", default=False, help="Enable query mode"
    )

    parser.addoption(
        "--metrics",
        action="append",
        default=None,
        required=False,
        choices=ALL_AVAILABLE_METRICS,
        help=(
            "Specify the metrics we want to benchmark. "
            "If not specified, the metric items will vary according to the specified operation's category and name."
        ),
    )

    parser.addoption(
        "--dtypes",
        action="append",
        default=None,
        required=False,
        choices=[str(ele) for ele in FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES],
        help=(
            "Specify the data types for benchmarks. "
            "If not specified, the dtype items will vary according to the specified operation's category and name."
        ),
    )

    parser.addoption(
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log"],
        help="Benchmark info recorded in log files or not",
    )


def pytest_configure(config):
    global Config
    mode_value = config.getoption("--mode")
    Config.cpu_mode = mode_value == "cpu"

    level_value = config.getoption("--level")
    Config.bench_level = BenchLevel(level_value)

    warmup_value = config.getoption("--warmup")
    Config.warm_up = int(warmup_value)

    iter_value = config.getoption("--iter")
    Config.repetition = int(iter_value)

    dtypes = config.getoption("--dtypes")
    Config.user_desired_dtypes = dtypes

    metrics = config.getoption("--metrics")
    Config.user_desired_metrics = metrics

    Config.record_log = config.getoption("--record") == "log"
    if Config.record_log:
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]
        logging.basicConfig(
            filename="result_{}.log".format("_".join(cmd_args)).replace("_-", "-"),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )


@pytest.fixture(scope="session", autouse=True)
def setup_once(request):
    if request.config.getoption("--query"):
        print("")
        print("This is query mode; skipping all real benchmark functions.")


@pytest.fixture(autouse=True)
def extract_and_log_op_attributes(request):
    print("")
    op_attributes = []

    # Extract the 'recommended_shapes' attribute from the pytest marker decoration.
    for mark in request.node.iter_markers():
        op_specified_shapes = mark.kwargs.get("recommended_shapes")
        rec_core_shapes = get_recommended_shapes(mark.name, op_specified_shapes)

        if rec_core_shapes:
            attri = OperationAttribute(
                op_name=mark.name,
                recommended_core_shapes=rec_core_shapes,
            )
            print(attri)
            op_attributes.append(attri.to_dict())

    if request.config.getoption("--query"):
        # Skip the real benchmark functions
        pytest.skip("Skipping benchmark due to the query parameter.")

    yield

    if Config.record_log and op_attributes:
        logging.info(json.dumps(op_attributes, indent=2))
