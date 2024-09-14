def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default="cuda",
        required=False,
        choices=["cuda", "cpu"],
        help="device to run reference tests on",
    )
    parser.addoption(
        "--shape",
        action="store",
        default="all",
        required=False,
        choices=["all", "one"],
        help="how many shapes to run tests on",
    )


def pytest_configure(config):
    global TO_CPU
    TO_CPU = config.getoption("--ref") == "cpu"

    global ONE_SHAPE
    ONE_SHAPE = config.getoption("--shape") == "one"
