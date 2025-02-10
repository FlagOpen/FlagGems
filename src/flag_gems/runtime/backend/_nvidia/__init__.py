<<<<<<< HEAD
from backend_utils import Autograd, VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="nvidia", device_name="cuda", device_query_cmd="nvidia-smi"
)

CUSTOMIZED_UNUSED_OPS = ("cumsum", "cos", "add")


__all__ = ["*", "HEURISTICS_CONFIGS"]
=======
from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="nvidia", device_name="cuda", device_query_cmd="nvidia-smi"
)

CUSTOMIZED_UNUSED_OPS = ("cumsum", "cos", "add")


__all__ = ["*"]
>>>>>>> 2aaa836 (SW-49945: Update configurations for multi-backends)
