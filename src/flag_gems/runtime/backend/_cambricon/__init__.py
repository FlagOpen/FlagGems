from backend_utils import VendorInfoBase  # noqa: E402

from .ops import *  # noqa: F403

vendor_info = VendorInfoBase(
    vendor_name="cambricon", device_name="mlu", device_query_cmd="cnmon"
)


def get_register_op_config():
    return ()
    return (("add.Tensor", add, False),)


def get_unused_op():
    return ()
    return ("cumsum", "cos")


__all__ = ["*"]
