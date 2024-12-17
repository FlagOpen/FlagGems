from backend_utils import VendorInfoBase  # noqa: E402

from .ops import *  # noqa: F403

vendor_info = VendorInfoBase(
        vendor_name="mthreads", device_name="musa", device_query_cmd="mthreads-gmi"
)


def get_register_op_config():
    # return (("add.Tensor", add, False),)
    return ()


def get_unused_op():
    # return ["cumsum", "cos"]
    return []


__all__ = ["*"]
