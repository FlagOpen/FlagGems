from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="aipu", device_name="aipu", device_query_cmd="aipu"
)

import triton
module = triton.runtime.driver.active.get_active_torch_device()

import sys
name = "torch.backends"
m = sys.modules[name]
setattr(m, "aipu", module)
torch_module_name = ".".join([name, "aipu"])
sys.modules[torch_module_name] = module

CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["*"]
