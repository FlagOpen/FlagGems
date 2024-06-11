def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="musa",
        required=False,
        choices=["musa", "cpu"],
        help="record latency in musa or cpu",
    )


def pytest_configure(config):
    value = config.getoption("--mode")
    global CPU_MODE
    CPU_MODE = value == "cpu"
