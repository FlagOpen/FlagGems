def pytest_addoption(parser):
    parser.addoption(
        "--fg_mode",
        action="store",
        default="cuda",
        required=False,
        choices=["cuda", "cpu"],
        help="record latency in cuda or cpu",
    )


def pytest_configure(config):
    value = config.getoption("--fg_mode")
    global CPU_MODE
    CPU_MODE = value == "cpu"
