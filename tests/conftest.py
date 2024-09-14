def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default="cuda",
        required=False,
        choices=["cuda", "cpu"],
        help="device to run reference tests on",
    )


def pytest_configure(config):
    value = config.getoption("--ref")
    global TO_CPU
    TO_CPU = value == "cpu"
