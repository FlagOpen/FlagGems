import os
import sys

from .ops import *  # noqa: F403

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend_utils import vendor_info_base  # noqa: E402

vendor_info = vendor_info_base(
    vendor_name="nvidia", device_name="cuda", cmd="nvidia-smi"
)


def get_register_op_config():
    return (("add.Tensor", add, False),)


def get_unused_op():
    return ("cumsum", "cos")


__all__ = ["*"]
