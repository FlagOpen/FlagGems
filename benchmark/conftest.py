import json
import logging

import pytest

from .attri_util import (
    DEFAULT_ITER_COUNT,
    DEFAULT_WARMUP_COUNT,
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
    Config.warm_up = warmup_value

    iter_value = config.getoption("--iter")
    Config.repetition = iter_value

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
def query_mode(request):
    print("")
    # Attempt to extract the 'recommended_shapes' attribute from the pytest marker decoration.
    op_attris = []
    for mark in request.node.iter_markers():
        op_specified_shapes = mark.kwargs.get("recommended_shapes")
        rec_core_shapes = get_recommended_shapes(mark.name, op_specified_shapes)
        if len(rec_core_shapes) != 0:
            attri = OperationAttribute(
                op_name=mark.name,
                recommended_core_shapes=rec_core_shapes,
            )
            print(attri)
            op_attris.append(attri.to_dict())

    if request.config.getoption("--query"):
        # Skip the real benchmark functions
        pytest.skip("Skipping benchmark due to the query param.")

    yield

    if Config.record_log and len(op_attris) > 0:
        logging.info(json.dumps(op_attris, indent=2))
