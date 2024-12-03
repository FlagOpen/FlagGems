from backend_utils import vendorInfoBase  # noqa: E402

from .ops import *  # noqa: F403

vendor_info = vendorInfoBase(vendor_name="nvidia", device_name="cuda", cmd="nvidia-smi")


def get_register_op_config():
    return (("add.Tensor", add, False),)


def get_unused_op():
    return ("cumsum", "cos")


__all__ = ["*"]
