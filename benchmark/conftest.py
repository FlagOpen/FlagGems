DEVICE = "mlu"
try:
    from torch_mlu.utils.model_transfer import transfer
except ImportError:
    DEVICE = "cuda"


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        action="store",
        default="mlu",
        required=False,
        choices=["cuda", "cpu", "mlu"],
        help="record latency in cuda or cpu or mlu",
    )


def pytest_configure(config):
    value = config.getoption("--mode")
    global CPU_MODE
    CPU_MODE = value == "cpu"
