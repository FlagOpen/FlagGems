import triton

from ..runtime import backend
from ..runtime.backend.device import DeviceDetector

device = DeviceDetector()
backend.set_torch_backend_device_fn(device.vendor_name)
tl_extra_shim = None
if tl_extra_shim is None:
    try:
        backend.set_tl_extra_backend_module(device.vendor_name)
        tl_extra_shim = backend.get_tl_extra_backend_module()
    except ImportError:
        try:
            tl_extra_shim = triton.language.math
        except ImportError:
            tl_extra_shim = triton.language.libdevice
