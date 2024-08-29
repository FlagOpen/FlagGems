DEVICE = "mlu"
try:
    from torch_mlu.utils.model_transfer import transfer
except ImportError:
    DEVICE = "cuda"


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="mlu",
        required=False,
        choices=["cuda", "cpu", "mlu"],
        help="device to run reference tests on",
    )


def pytest_configure(config):
    value = config.getoption("--device")
    global TO_CPU
    TO_CPU = value == "cpu"
