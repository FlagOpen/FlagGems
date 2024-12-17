from backend_utils import VendorInfoBase  # noqa: E402

from .ops import *  # noqa: F403

vendor_info = VendorInfoBase(
    vendor_name="metax", device_name="cuda", device_query_cmd="mx-smi"
)


def get_register_ops():
    return (("add.Tensor", add, False),)


def get_unused_op():
    return []

def get_register_op_config():
    return ()


__all__ = ["*"]
